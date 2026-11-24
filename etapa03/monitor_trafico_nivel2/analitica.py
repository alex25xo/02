"""
NIVEL 2 - Regenerar o COMBINAR la analítica de tránsito.

El radar ya genera los gráficos automáticamente al cerrarse. Este script sirve para:
  - volver a generarlos (por ejemplo, forzando el número de carriles)
  - JUNTAR varios clips grabados a distintas horas desde el mismo punto

Uso:
    python analitica.py salidas/mi_video
    python analitica.py salidas/mi_video --carriles 4
    python analitica.py salidas/clip_0700 salidas/clip_1200 salidas/clip_1800 --salida salidas/dia_completo
"""
import argparse
import sys

from radar.analitica import generar_reportes


def main():
    ap = argparse.ArgumentParser(description="Analitica de transito (Nivel 2)")
    ap.add_argument("carpetas", nargs="+", help="Carpetas de salidas/ generadas por el radar")
    ap.add_argument("--salida", help="Carpeta destino (por defecto la primera carpeta)")
    ap.add_argument("--carriles", type=int, default=None,
                    help="Numero de carriles (por defecto se detectan automaticamente)")
    ap.add_argument("--carriles-def", default=None,
                    help="Carriles dibujados a mano con definir_carriles.py")
    args = ap.parse_args()
    try:
        r = generar_reportes(args.carpetas, args.salida, args.carriles, carriles_def=args.carriles_def)
    except FileNotFoundError as e:
        sys.exit(str(e))
    print("\nGraficos generados:")
    for ruta in r["archivos"]:
        print(f"  - {ruta}")


if __name__ == "__main__":
    main()
