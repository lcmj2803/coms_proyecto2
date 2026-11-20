"""Arquitectura CNN 1D para clasificar señales I/Q."""

from torch import nn


class ModulationCNN(nn.Module):
    def __init__(self, num_classes=7):
        super(ModulationCNN, self).__init__()

        self.features = nn.Sequential(
            # Entrada: (lote, 2, 1024).
            # Los dos canales son I y Q.
            nn.Conv1d(
                in_channels=2,
                out_channels=16,
                kernel_size=7,
                padding=3
            ),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            # Forma: (lote, 16, 512).
            nn.Conv1d(
                in_channels=16,
                out_channels=32,
                kernel_size=5,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            # Forma: (lote, 32, 256).
            nn.Conv1d(
                in_channels=32,
                out_channels=64,
                kernel_size=5,
                padding=2
            ),
            nn.ReLU(),

            # Promedio temporal de cada mapa.
            # Salida: (lote, 64, 1).
            nn.AdaptiveAvgPool1d(1)
        )

        # Una puntuación por clase.
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        """Transformar un lote I/Q en puntuaciones por clase."""
        x = self.features(x)

        # Eliminar solo la última dimensión:
        # (lote, 64, 1) -> (lote, 64).
        x = x.squeeze(-1)

        # Salida: (lote, num_classes).
        logits = self.classifier(x)

        return logits
    
