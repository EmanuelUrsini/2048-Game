import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import random
import numpy as np
import json
import time
import threading
from collections import deque
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
from abc import ABC, abstractmethod

# Configuraciones visuales avanzadas
THEMES = {
    "Clásico": {
        "background": "#bbada0",
        "empty": "#cdc1b4",
        "cell_colors": {
            0: "#cdc1b4",
            2: "#eee4da", 4: "#ede0c8",
            8: "#f2b179", 16: "#f59563",
            32: "#f67c5f", 64: "#f65e3b",
            128: "#edcf72", 256: "#edcc61",
            512: "#edc850", 1024: "#edc53f",
            2048: "#edc22e"
        }
    },
    "Oscuro": {
        "background": "#2d2d2d",
        "empty": "#3c3c3c",
        "cell_colors": {
            0: "#3c3c3c",
            2: "#4a4a4a", 4: "#5a5a5a",
            8: "#6c6c6c", 16: "#7d7d7d",
            32: "#8f8f8f", 64: "#a1a1a1",
            128: "#b3b3b3", 256: "#c5c5c5",
            512: "#d7d7d7", 1024: "#e9e9e9",
            2048: "#ffffff"
        }
    }
}


@dataclass
class GameResult:
    """Resultado de una partida individual"""
    puntuacion: int
    movimientos: int
    max_tile: int
    tiempo_ejecucion: float
    tablero_final: np.ndarray
    estrategia: str
    timestamp: str


@dataclass
class ExperimentConfig:
    """Configuración de experimento"""
    num_simulaciones: int = 1000
    semilla_aleatoria: int = 42
    timeout_partida: float = 300.0
    profundidad_expectiminimax: int = 3
    guardar_tableros_finales: bool = False


class EstrategiaIA(ABC):
    """Clase abstracta base para estrategias de IA"""

    @abstractmethod
    def obtener_movimiento(self, tablero: np.ndarray) -> Optional[str]:
        """Obtiene el próximo movimiento basado en el estado del tablero"""
        pass

    @property
    @abstractmethod
    def nombre(self) -> str:
        """Nombre identificativo de la estrategia"""
        pass


class EstrategiaAleatoria(EstrategiaIA):
    """Estrategia que selecciona movimientos aleatoriamente"""

    def obtener_movimiento(self, tablero: np.ndarray) -> Optional[str]:
        movimientos_posibles = ['Up', 'Down', 'Left', 'Right']
        return random.choice(movimientos_posibles)

    @property
    def nombre(self) -> str:
        return "Aleatoria"


class EstrategiaPrioridadDerecha(EstrategiaIA):
    """Estrategia que prioriza movimientos hacia la derecha"""

    def obtener_movimiento(self, tablero: np.ndarray) -> Optional[str]:
        # Prioridad: Derecha -> Abajo -> Arriba -> Izquierda
        for movimiento in ['Right', 'Down', 'Up', 'Left']:
            # Simular el movimiento para verificar si es válido
            tablero_temp = tablero.copy()
            if self._simular_movimiento(tablero_temp, movimiento):
                return movimiento
        return None

    def _simular_movimiento(self, tablero: np.ndarray, movimiento: str) -> bool:
        """Simula un movimiento para verificar si es válido"""
        tablero_original = tablero.copy()

        if movimiento == 'Right':
            for i in range(4):
                row = tablero[i, :][::-1]
                new_row, _ = self._slide_and_merge_sim(row)
                tablero[i, :] = new_row[::-1]
        elif movimiento == 'Left':
            for i in range(4):
                row = tablero[i, :]
                new_row, _ = self._slide_and_merge_sim(row)
                tablero[i, :] = new_row
        elif movimiento == 'Down':
            for j in range(4):
                col = tablero[:, j][::-1]
                new_col, _ = self._slide_and_merge_sim(col)
                tablero[:, j] = new_col[::-1]
        elif movimiento == 'Up':
            for j in range(4):
                col = tablero[:, j]
                new_col, _ = self._slide_and_merge_sim(col)
                tablero[:, j] = new_col

        return not np.array_equal(tablero_original, tablero)

    def _slide_and_merge_sim(self, line: np.ndarray) -> tuple:
        """Simulación de slide_and_merge sin modificar el objeto game"""
        non_zero = line[line != 0]
        new_line = np.zeros_like(line)
        score_add = 0
        idx = 0

        i = 0
        while i < len(non_zero):
            if i + 1 < len(non_zero) and non_zero[i] == non_zero[i + 1]:
                merged_value = non_zero[i] * 2
                new_line[idx] = merged_value
                score_add += merged_value
                i += 2
            else:
                new_line[idx] = non_zero[i]
                i += 1
            idx += 1

        return new_line, score_add

    @property
    def nombre(self) -> str:
        return "Prioridad Derecha"


class EstrategiaOptimal2048(EstrategiaIA):
    """Estrategia Expectiminimax optimizada para 2048"""

    def __init__(self, profundidad: int = 3):
        self.profundidad = profundidad
        self.pesos = {
            'monotonicidad': 1.0,
            'suavidad': 0.1,
            'vacias': 2.7,
            'max_esquina': 1.0
        }

    def obtener_movimiento(self, tablero: np.ndarray) -> Optional[str]:
        mejor_puntuacion = -float('inf')
        mejor_movimiento = None

        for movimiento in ['Right', 'Down', 'Left', 'Up']:
            tablero_temp = tablero.copy()
            if self._simular_movimiento(tablero_temp, movimiento):
                puntuacion = self._expectiminimax(tablero_temp, self.profundidad - 1, False)
                if puntuacion > mejor_puntuacion:
                    mejor_puntuacion = puntuacion
                    mejor_movimiento = movimiento

        return mejor_movimiento

    def _expectiminimax(self, tablero: np.ndarray, profundidad: int, es_jugador: bool) -> float:
        """Algoritmo Expectiminimax con poda alfa-beta"""
        if profundidad == 0:
            return self._evaluar_tablero(tablero)

        if es_jugador:
            # Turno del jugador (maximizar)
            max_eval = -float('inf')
            for movimiento in ['Right', 'Down', 'Left', 'Up']:
                tablero_temp = tablero.copy()
                if self._simular_movimiento(tablero_temp, movimiento):
                    eval_actual = self._expectiminimax(tablero_temp, profundidad - 1, False)
                    max_eval = max(max_eval, eval_actual)
            return max_eval if max_eval != -float('inf') else 0
        else:
            # Turno del entorno (valor esperado)
            celdas_vacias = [(i, j) for i in range(4) for j in range(4) if tablero[i][j] == 0]
            if not celdas_vacias:
                return self._evaluar_tablero(tablero)

            total = 0
            for (i, j) in celdas_vacias:
                # Ficha 2 (probabilidad 0.9)
                tablero[i][j] = 2
                total += 0.9 * self._expectiminimax(tablero, profundidad - 1, True)

                # Ficha 4 (probabilidad 0.1)
                tablero[i][j] = 4
                total += 0.1 * self._expectiminimax(tablero, profundidad - 1, True)

                tablero[i][j] = 0

            return total / len(celdas_vacias)

    def _evaluar_tablero(self, tablero: np.ndarray) -> float:
        """Función de evaluación heurística multifactorial"""
        monotonicidad = self._calcular_monotonicidad(tablero)
        suavidad = self._calcular_suavidad(tablero)
        vacias = np.count_nonzero(tablero == 0)
        max_esquina = self._calcular_bonus_esquina(tablero)

        return (self.pesos['monotonicidad'] * monotonicidad +
                self.pesos['suavidad'] * suavidad +
                self.pesos['vacias'] * vacias +
                self.pesos['max_esquina'] * max_esquina)

    def _calcular_monotonicidad(self, tablero: np.ndarray) -> float:
        """Calcula la monotonicidad del tablero"""
        mono = 0
        for i in range(4):
            mono += self._monotonicity_row(tablero[i, :])
            mono += self._monotonicity_row(tablero[:, i])
        return mono

    def _monotonicity_row(self, row: np.ndarray) -> float:
        """Calcula monotonicidad de una fila/columna"""
        increases = decreases = 0
        for i in range(len(row) - 1):
            if row[i] > row[i + 1]:
                decreases += row[i] - row[i + 1]
            else:
                increases += row[i + 1] - row[i]
        return max(increases, decreases)

    def _calcular_suavidad(self, tablero: np.ndarray) -> float:
        """Calcula la suavidad (diferencias entre celdas adyacentes)"""
        suavidad = 0
        for i in range(4):
            for j in range(4):
                if j < 3:  # Derecha
                    suavidad -= abs(tablero[i][j] - tablero[i][j + 1])
                if i < 3:  # Abajo
                    suavidad -= abs(tablero[i][j] - tablero[i + 1][j])
        return suavidad

    def _calcular_bonus_esquina(self, tablero: np.ndarray) -> float:
        """Calcula bonus por mantener ficha máxima en esquina"""
        max_tile = np.max(tablero)
        max_pos = np.unravel_index(tablero.argmax(), tablero.shape)

        # Bonus por esquinas (prioridad esquina inferior derecha)
        esquinas = [(0, 0), (0, 3), (3, 0), (3, 3)]
        if max_pos in esquinas:
            if max_pos == (3, 3):  # Esquina preferida
                return max_tile * 0.7
            else:
                return max_tile * 0.5
        return 0

    def _simular_movimiento(self, tablero: np.ndarray, movimiento: str) -> bool:
        """Simula un movimiento y retorna si fue exitoso"""
        tablero_original = tablero.copy()

        if movimiento == 'Right':
            for i in range(4):
                row = tablero[i, :][::-1]
                new_row, _ = self._slide_and_merge_sim(row)
                tablero[i, :] = new_row[::-1]
        elif movimiento == 'Left':
            for i in range(4):
                row = tablero[i, :]
                new_row, _ = self._slide_and_merge_sim(row)
                tablero[i, :] = new_row
        elif movimiento == 'Down':
            for j in range(4):
                col = tablero[:, j][::-1]
                new_col, _ = self._slide_and_merge_sim(col)
                tablero[:, j] = new_col[::-1]
        elif movimiento == 'Up':
            for j in range(4):
                col = tablero[:, j]
                new_col, _ = self._slide_and_merge_sim(col)
                tablero[:, j] = new_col

        return not np.array_equal(tablero_original, tablero)

    def _slide_and_merge_sim(self, line: np.ndarray) -> tuple:
        """Simulación de slide_and_merge"""
        non_zero = line[line != 0]
        new_line = np.zeros_like(line)
        score_add = 0
        idx = 0

        i = 0
        while i < len(non_zero):
            if i + 1 < len(non_zero) and non_zero[i] == non_zero[i + 1]:
                merged_value = non_zero[i] * 2
                new_line[idx] = merged_value
                score_add += merged_value
                i += 2
            else:
                new_line[idx] = non_zero[i]
                i += 1
            idx += 1

        return new_line, score_add

    @property
    def nombre(self) -> str:
        return "Optimal 2048"


class SimuladorMonteCarlo:
    """Sistema de simulación Monte Carlo para evaluación de estrategias"""

    def __init__(self, config: ExperimentConfig = ExperimentConfig()):
        self.config = config
        self.resultados = []
        self.ejecutando = False

    def ejecutar_experimento(self, estrategias: List[EstrategiaIA],
                             callback_progreso=None) -> Dict[str, List[GameResult]]:
        """
        Ejecuta experimento completo con múltiples estrategias

        Args:
            estrategias: Lista de estrategias a evaluar
            callback_progreso: Función para reportar progreso (opcional)

        Returns:
            Diccionario con resultados por estrategia
        """
        random.seed(self.config.semilla_aleatoria)
        np.random.seed(self.config.semilla_aleatoria)

        self.ejecutando = True
        resultados_experimento = {}

        total_simulaciones = len(estrategias) * self.config.num_simulaciones
        simulacion_actual = 0

        for estrategia in estrategias:
            print(f"\n🎮 Ejecutando {self.config.num_simulaciones} simulaciones para {estrategia.nombre}")
            resultados_estrategia = []

            for i in range(self.config.num_simulaciones):
                if not self.ejecutando:  # Permitir cancelación
                    break

                resultado = self._simular_partida(estrategia)
                resultados_estrategia.append(resultado)

                simulacion_actual += 1

                # Reportar progreso
                if callback_progreso:
                    progreso = (simulacion_actual / total_simulaciones) * 100
                    callback_progreso(progreso, estrategia.nombre, i + 1)

                if (i + 1) % 100 == 0:
                    print(f"  ✅ Progreso: {i + 1}/{self.config.num_simulaciones}")

            resultados_experimento[estrategia.nombre] = resultados_estrategia

            # Mostrar estadísticas preliminares
            self._mostrar_estadisticas_preliminares(estrategia.nombre, resultados_estrategia)

        self.resultados = resultados_experimento
        return resultados_experimento

    def _simular_partida(self, estrategia: EstrategiaIA) -> GameResult:
        """Simula una partida individual"""
        tiempo_inicio = time.time()
        juego = Game2048Engine()
        juego.reset_game()

        while not juego.is_game_over():
            movimiento = estrategia.obtener_movimiento(juego.board)
            if movimiento:
                moved = juego.ejecutar_movimiento(movimiento)
                if moved:
                    juego.add_random_tile()
            else:
                break

        tiempo_total = time.time() - tiempo_inicio

        # Obtener el valor máximo del tablero final
        final_max = np.max(juego.board)

        return GameResult(
            puntuacion=juego.score,
            movimientos=juego.move_count,
            max_tile=final_max,
            tiempo_ejecucion=tiempo_total,
            tablero_final=juego.board.copy() if self.config.guardar_tableros_finales else None,
            estrategia=estrategia.nombre,
            timestamp=datetime.now().isoformat()
        )

    def _mostrar_estadisticas_preliminares(self, nombre_estrategia: str, resultados: List[GameResult]):
        """Muestra estadísticas básicas de una estrategia"""
        puntuaciones = [r.puntuacion for r in resultados]

        print(f"  📊 {nombre_estrategia}:")
        print(f"     Puntuación promedio: {np.mean(puntuaciones):.0f}")
        print(f"     Puntuación máxima: {np.max(puntuaciones)}")
        print(f"     Tiempo promedio: {np.mean([r.tiempo_ejecucion for r in resultados]):.2f}s")

    def detener_experimento(self):
        """Detiene la ejecución del experimento"""
        self.ejecutando = False

    def generar_reporte_completo(self) -> str:
        """Genera reporte estadístico completo"""
        if not self.resultados:
            return "No hay resultados para generar reporte"

        reporte = ["=" * 80]
        reporte.append("🎯 REPORTE ESTADÍSTICO COMPLETO - SIMULACIÓN MONTE CARLO 2048")
        reporte.append("=" * 80)
        reporte.append(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        reporte.append(f"Simulaciones por estrategia: {self.config.num_simulaciones}")
        reporte.append(f"Semilla aleatoria: {self.config.semilla_aleatoria}")
        reporte.append("")

        # Estadísticas por estrategia
        for estrategia, resultados in self.resultados.items():
            reporte.extend(self._generar_estadisticas_estrategia(estrategia, resultados))

        # Análisis comparativo
        reporte.extend(self._generar_analisis_comparativo())

        return "\n".join(reporte)

    def _generar_estadisticas_estrategia(self, nombre: str, resultados: List[GameResult]) -> List[str]:
        """Genera estadísticas detalladas para una estrategia"""
        puntuaciones = [r.puntuacion for r in resultados]
        movimientos = [r.movimientos for r in resultados]
        tiempos = [r.tiempo_ejecucion for r in resultados]

        stats_section = [
            f"\n📈 ESTRATEGIA: {nombre.upper()}",
            "-" * 50,
            f"Puntuación:",
            f"  Media: {np.mean(puntuaciones):.0f} ± {np.std(puntuaciones):.0f}",
            f"  Mediana: {np.median(puntuaciones):.0f}",
            f"  Rango: {np.min(puntuaciones)} - {np.max(puntuaciones)}",
            f"  Q1-Q3: {np.percentile(puntuaciones, 25):.0f} - {np.percentile(puntuaciones, 75):.0f}",
            f"",
            f"Movimientos:",
            f"  Promedio: {np.mean(movimientos):.0f} ± {np.std(movimientos):.0f}",
            f"  Rango: {np.min(movimientos)} - {np.max(movimientos)}",
            f"",
            f"Tiempo:",
            f"  Promedio: {np.mean(tiempos):.3f}s",
            f"",
            f"Distribución de fichas máximas:",
        ]

        # Distribución de fichas máximas
        max_tiles = [r.max_tile for r in resultados]
        tiles_unicos = sorted(set(max_tiles))
        for tile in tiles_unicos:
            count = sum(1 for t in max_tiles if t == tile)
            porcentaje = count / len(max_tiles) * 100
            stats_section.append(f"  {tile}: {count} partidas ({porcentaje:.1f}%)")

        return stats_section

    def _generar_analisis_comparativo(self) -> List[str]:
        """Genera análisis estadístico comparativo entre estrategias"""
        if len(self.resultados) < 2:
            return ["\n⚠️  Se necesitan al menos 2 estrategias para análisis comparativo"]

        analisis = [
            "\n🔍 ANÁLISIS COMPARATIVO",
            "=" * 50,
        ]

        # Preparar datos
        datos_estrategias = {}
        for estrategia, resultados in self.resultados.items():
            datos_estrategias[estrategia] = {
                'puntuaciones': [r.puntuacion for r in resultados],
                'movimientos': [r.movimientos for r in resultados],
                'tiempos': [r.tiempo_ejecucion for r in resultados]
            }

        # Ranking por puntuación promedio
        ranking = sorted(datos_estrategias.items(),
                         key=lambda x: np.mean(x[1]['puntuaciones']),
                         reverse=True)

        analisis.append("\n🏆 RANKING POR PUNTUACIÓN PROMEDIO:")
        for i, (estrategia, datos) in enumerate(ranking, 1):
            puntuacion_media = np.mean(datos['puntuaciones'])
            analisis.append(f"{i}. {estrategia}: {puntuacion_media:.0f} puntos")

        # Análisis de significancia estadística (si scipy está disponible)
        try:
            analisis.extend(self._realizar_pruebas_estadisticas(datos_estrategias))
        except ImportError:
            analisis.append("\n⚠️  Scipy no disponible para pruebas estadísticas avanzadas")

        # Balance eficacia-eficiencia
        analisis.append("\n⚖️  BALANCE EFICACIA-EFICIENCIA:")
        for estrategia, datos in datos_estrategias.items():
            eficacia = np.mean(datos['puntuaciones'])
            eficiencia = 1 / np.mean(datos['tiempos'])  # Inverso del tiempo
            ratio = eficacia / np.mean(datos['tiempos'])
            analisis.append(f"{estrategia}: {ratio:.0f} puntos/segundo")

        return analisis

    def _realizar_pruebas_estadisticas(self, datos_estrategias: Dict) -> List[str]:
        """Realiza pruebas estadísticas entre estrategias"""
        resultados_stats = ["\n📊 PRUEBAS ESTADÍSTICAS:"]

        estrategias = list(datos_estrategias.keys())
        puntuaciones_grupos = [datos_estrategias[e]['puntuaciones'] for e in estrategias]

        # ANOVA de una vía
        if len(estrategias) > 2:
            f_stat, p_valor = stats.f_oneway(*puntuaciones_grupos)
            resultados_stats.append(f"\nANOVA de una vía:")
            resultados_stats.append(f"  F-estadístico: {f_stat:.2f}")
            resultados_stats.append(f"  p-valor: {p_valor:.6f}")
            if p_valor < 0.05:
                resultados_stats.append("  ✅ Diferencias significativas entre estrategias (p < 0.05)")
            else:
                resultados_stats.append("  ❌ No hay diferencias significativas (p ≥ 0.05)")

        # Pruebas t entre pares
        resultados_stats.append(f"\nPruebas t de Student (pares):")
        for i in range(len(estrategias)):
            for j in range(i + 1, len(estrategias)):
                t_stat, p_val = stats.ttest_ind(puntuaciones_grupos[i], puntuaciones_grupos[j])
                significativo = "✅" if p_val < 0.05 else "❌"
                resultados_stats.append(
                    f"  {estrategias[i]} vs {estrategias[j]}: "
                    f"t={t_stat:.2f}, p={p_val:.4f} {significativo}"
                )

        return resultados_stats

    def exportar_datos_csv(self, archivo: str = None) -> str:
        """Exporta resultados a CSV"""
        if not archivo:
            archivo = f"resultados_2048_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

        # Convertir resultados a lista plana
        datos_csv = []
        for estrategia, resultados in self.resultados.items():
            for resultado in resultados:
                datos_csv.append({
                    'estrategia': resultado.estrategia,
                    'puntuacion': resultado.puntuacion,
                    'movimientos': resultado.movimientos,
                    'max_tile': resultado.max_tile,
                    'tiempo_ejecucion': resultado.tiempo_ejecucion,
                    'timestamp': resultado.timestamp
                })

        # Crear DataFrame y exportar
        try:
            df = pd.DataFrame(datos_csv)
            df.to_csv(archivo, index=False)
            return f"✅ Datos exportados a: {archivo}"
        except ImportError:
            # Exportar manualmente si pandas no está disponible
            with open(archivo, 'w') as f:
                f.write("estrategia,puntuacion,movimientos,max_tile,tiempo_ejecucion,timestamp\n")
                for dato in datos_csv:
                    f.write(f"{dato['estrategia']},{dato['puntuacion']},{dato['movimientos']},"
                            f"{dato['max_tile']},{dato['tiempo_ejecucion']},"
                            f"{dato['timestamp']}\n")
            return f"✅ Datos exportados a: {archivo} (formato manual)"


class Game2048Engine:
    """Motor del juego 2048 sin interfaz gráfica para simulaciones"""

    def __init__(self):
        self.grid_size = 4
        self.board = np.zeros((self.grid_size, self.grid_size), dtype=int)
        self.score = 0
        self.move_count = 0
        self.max_tile = 0
        self.game_over = False

    def reset_game(self):
        """Reinicia el juego"""
        self.board = np.zeros((self.grid_size, self.grid_size), dtype=int)
        self.score = 0
        self.move_count = 0
        self.max_tile = 0
        self.game_over = False
        self.add_random_tile()
        self.add_random_tile()

    def add_random_tile(self):
        """Añade una ficha aleatoria (2 o 4) en posición vacía"""
        empty_cells = [(i, j) for i in range(self.grid_size) for j in range(self.grid_size)
                       if self.board[i][j] == 0]
        if empty_cells:
            i, j = random.choice(empty_cells)
            new_value = 2 if random.random() < 0.9 else 4
            self.board[i][j] = new_value
            self.max_tile = max(self.max_tile, new_value)

    def ejecutar_movimiento(self, movimiento: str) -> bool:
        """Ejecuta un movimiento y retorna si fue exitoso"""
        if movimiento == 'Up':
            return self.move_up()
        elif movimiento == 'Down':
            return self.move_down()
        elif movimiento == 'Left':
            return self.move_left()
        elif movimiento == 'Right':
            return self.move_right()
        return False

    def move_up(self):
        """Movimiento hacia arriba"""
        moved = False
        for j in range(self.grid_size):
            column = self.board[:, j]
            new_column, score_add = self.slide_and_merge(column)
            if not np.array_equal(column, new_column):
                self.board[:, j] = new_column
                self.score += score_add
                moved = True
        if moved:
            self.move_count += 1
        return moved

    def move_down(self):
        """Movimiento hacia abajo"""
        moved = False
        for j in range(self.grid_size):
            column = self.board[:, j][::-1]
            new_column, score_add = self.slide_and_merge(column)
            if not np.array_equal(column, new_column):
                self.board[:, j] = new_column[::-1]
                self.score += score_add
                moved = True
        if moved:
            self.move_count += 1
        return moved

    def move_left(self):
        """Movimiento hacia la izquierda"""
        moved = False
        for i in range(self.grid_size):
            row = self.board[i, :]
            new_row, score_add = self.slide_and_merge(row)
            if not np.array_equal(row, new_row):
                self.board[i, :] = new_row
                self.score += score_add
                moved = True
        if moved:
            self.move_count += 1
        return moved

    def move_right(self):
        """Movimiento hacia la derecha"""
        moved = False
        for i in range(self.grid_size):
            row = self.board[i, :][::-1]
            new_row, score_add = self.slide_and_merge(row)
            if not np.array_equal(row, new_row):
                self.board[i, :] = new_row[::-1]
                self.score += score_add
                moved = True
        if moved:
            self.move_count += 1
        return moved

    def slide_and_merge(self, line):
        """Desliza y fusiona una línea"""
        non_zero = line[line != 0]
        new_line = np.zeros_like(line)
        score_add = 0
        idx = 0
        new_max = self.max_tile

        i = 0
        while i < len(non_zero):
            if i + 1 < len(non_zero) and non_zero[i] == non_zero[i + 1]:
                merged_value = non_zero[i] * 2
                new_line[idx] = merged_value
                score_add += merged_value

                if merged_value > new_max:
                    new_max = merged_value
                i += 2
            else:
                new_line[idx] = non_zero[i]
                i += 1
            idx += 1

        if new_max > self.max_tile:
            self.max_tile = new_max

        return new_line, score_add

    def is_game_over(self):
        """Verifica si el juego ha terminado"""
        if 0 in self.board:
            return False

        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if j + 1 < self.grid_size and self.board[i][j] == self.board[i][j + 1]:
                    return False
                if i + 1 < self.grid_size and self.board[i][j] == self.board[i + 1][j]:
                    return False

        return True


class Game2048(Game2048Engine):
    """Clase principal del juego 2048 con interfaz gráfica (hereda del motor)"""

    def __init__(self, master):
        super().__init__()  # Inicializar motor del juego
        self.master = master
        self.master.title("2048 Advanced - Simulación Monte Carlo")

        # Configuración visual
        self.cell_size = 100
        self.padding = 10
        self.current_theme = THEMES["Clásico"]
        self.high_score = 0

        # Estado de simulación
        self.auto_playing = False
        self.current_strategy = None
        self.move_delay = 100
        self.simulador = None
        self.experimento_ejecutandose = False

        self.history = []  # Almacena estados anteriores

        # Fuentes de color para texto
        self.font_colors = {
            2: "#776e65", 4: "#776e65",
            8: "#f9f6f2", 16: "#f9f6f2",
            32: "#f9f6f2", 64: "#f9f6f2",
            128: "#f9f6f2", 256: "#f9f6f2",
            512: "#f9f6f2", 1024: "#f9f6f2",
            2048: "#f9f6f2"
        }

        # Interfaz gráfica
        self.setup_gui()
        self.reset_game()

    def setup_gui(self):
        # Frame principal
        self.main_frame = ttk.Frame(self.master)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # Configurar estilos ttk
        self.style = ttk.Style()
        self.style.configure("Icon.TButton", font=("Arial", 12))

        # Notebook para pestañas
        self.notebook = ttk.Notebook(self.main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Pestaña 1: Juego Manual/Auto
        self.setup_game_tab()

        # Pestaña 2: Simulación Monte Carlo
        self.setup_simulation_tab()

        # Pestaña 3: Análisis de Resultados
        self.setup_analysis_tab()

    def setup_game_tab(self):
        """Configura la pestaña de juego"""
        game_frame = ttk.Frame(self.notebook)
        self.notebook.add(game_frame, text="🎮 Juego")

        # Barra superior de controles
        control_frame = ttk.Frame(game_frame)
        control_frame.pack(fill=tk.X, padx=10, pady=5)

        # Selector de estrategia
        self.strategy_var = tk.StringVar()
        self.strategy_menu = ttk.Combobox(
            control_frame,
            textvariable=self.strategy_var,
            values=["Manual", "Aleatoria", "Prioridad Derecha", "Optimal 2048"],
            state="readonly",
            width=20
        )
        self.strategy_menu.pack(side=tk.LEFT, padx=5)
        self.strategy_menu.set("Manual")

        # Control de velocidad
        ttk.Label(control_frame, text="Velocidad:").pack(side=tk.LEFT, padx=5)
        self.speed_scale = ttk.Scale(
            control_frame,
            from_=50,
            to=500,
            command=lambda v: setattr(self, 'move_delay', int(float(v)))
        )
        self.speed_scale.set(self.move_delay)
        self.speed_scale.pack(side=tk.LEFT, padx=5)

        # Botones de control
        ttk.Button(
            control_frame,
            text="↻ Reiniciar",
            command=self.reset_game,
            style="Icon.TButton"
        ).pack(side=tk.LEFT, padx=2)

        ttk.Button(
            control_frame,
            text="⏮ Deshacer",
            command=self.undo_move,
            style="Icon.TButton"
        ).pack(side=tk.LEFT, padx=2)

        # Botón para jugar automático
        self.btn_auto = ttk.Button(
            control_frame,
            text="▶ Jugar Automático",
            command=self.toggle_auto_play
        )
        self.btn_auto.pack(side=tk.LEFT, padx=5)

        # Panel de información
        info_frame = ttk.Frame(game_frame)
        info_frame.pack(fill=tk.X, padx=10, pady=5)

        self.score_label = ttk.Label(info_frame, text=f"Puntuación: {self.score}", font=("Arial", 12, "bold"))
        self.score_label.pack(side=tk.LEFT, padx=10)

        self.high_score_label = ttk.Label(info_frame, text=f"Récord: {self.high_score}", font=("Arial", 12, "bold"))
        self.high_score_label.pack(side=tk.LEFT, padx=10)

        self.move_label = ttk.Label(info_frame, text=f"Movimientos: {self.move_count}", font=("Arial", 12, "bold"))
        self.move_label.pack(side=tk.LEFT, padx=10)

        self.max_tile_label = ttk.Label(info_frame, text=f"Máxima ficha: {self.max_tile}", font=("Arial", 12, "bold"))
        self.max_tile_label.pack(side=tk.LEFT, padx=10)

        # Tablero de juego
        self.canvas = tk.Canvas(
            game_frame,
            width=self.grid_size * self.cell_size + (self.grid_size + 1) * self.padding,
            height=self.grid_size * self.cell_size + (self.grid_size + 1) * self.padding,
            bg=self.current_theme["background"],
            highlightthickness=0
        )
        self.canvas.pack(padx=10, pady=10)

        # Menú de temas
        theme_menu = ttk.Frame(game_frame)
        theme_menu.pack(pady=5)
        for theme_name in THEMES:
            ttk.Button(
                theme_menu,
                text=theme_name,
                command=lambda tn=theme_name: self.change_theme(tn)
            ).pack(side=tk.LEFT, padx=5)

        # Bindings de teclado
        self.master.bind('<Key>', self.key_press)
        self.master.focus_set()

    def setup_simulation_tab(self):
        """Configura la pestaña de simulación Monte Carlo"""
        sim_frame = ttk.Frame(self.notebook)
        self.notebook.add(sim_frame, text="🧪 Simulación Monte Carlo")

        # Configuración del experimento
        config_frame = ttk.LabelFrame(sim_frame, text="⚙️ Configuración del Experimento")
        config_frame.pack(fill=tk.X, padx=10, pady=5)

        # Número de simulaciones
        ttk.Label(config_frame, text="Simulaciones por estrategia:").grid(row=0, column=0, sticky="w", padx=5, pady=2)
        self.sim_count_var = tk.StringVar(value="1000")
        sim_count_spinbox = ttk.Spinbox(config_frame, from_=10, to=10000, textvariable=self.sim_count_var, width=10)
        sim_count_spinbox.grid(row=0, column=1, padx=5, pady=2)

        # Semilla aleatoria
        ttk.Label(config_frame, text="Semilla aleatoria:").grid(row=1, column=0, sticky="w", padx=5, pady=2)
        self.seed_var = tk.StringVar(value="42")
        seed_entry = ttk.Entry(config_frame, textvariable=self.seed_var, width=10)
        seed_entry.grid(row=1, column=1, padx=5, pady=2)

        # Profundidad Expectiminimax
        ttk.Label(config_frame, text="Profundidad Expectiminimax:").grid(row=2, column=0, sticky="w", padx=5, pady=2)
        self.depth_var = tk.StringVar(value="3")
        depth_spinbox = ttk.Spinbox(config_frame, from_=1, to=5, textvariable=self.depth_var, width=10)
        depth_spinbox.grid(row=2, column=1, padx=5, pady=2)

        # Selección de estrategias
        strategies_frame = ttk.LabelFrame(sim_frame, text="🎯 Estrategias a Evaluar")
        strategies_frame.pack(fill=tk.X, padx=10, pady=5)

        self.strategy_vars = {}
        strategies = ["Aleatoria", "Prioridad Derecha", "Optimal 2048"]
        for i, strategy in enumerate(strategies):
            var = tk.BooleanVar(value=True)
            self.strategy_vars[strategy] = var
            ttk.Checkbutton(strategies_frame, text=strategy, variable=var).grid(row=0, column=i, padx=10, pady=5)

        # Control de simulación
        control_sim_frame = ttk.Frame(sim_frame)
        control_sim_frame.pack(fill=tk.X, padx=10, pady=5)

        self.btn_start_sim = ttk.Button(
            control_sim_frame,
            text="🚀 Ejecutar Simulación",
            command=self.iniciar_simulacion_threaded
        )
        self.btn_start_sim.pack(side=tk.LEFT, padx=5)

        self.btn_stop_sim = ttk.Button(
            control_sim_frame,
            text="⏹ Detener",
            command=self.detener_simulacion,
            state=tk.DISABLED
        )
        self.btn_stop_sim.pack(side=tk.LEFT, padx=5)

        ttk.Button(
            control_sim_frame,
            text="💾 Exportar CSV",
            command=self.exportar_resultados
        ).pack(side=tk.LEFT, padx=5)

        # Barra de progreso
        progress_frame = ttk.Frame(sim_frame)
        progress_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(progress_frame, text="Progreso:").pack(side=tk.LEFT)
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            progress_frame,
            variable=self.progress_var,
            maximum=100,
            length=300
        )
        self.progress_bar.pack(side=tk.LEFT, padx=10, fill=tk.X, expand=True)

        self.progress_label = ttk.Label(progress_frame, text="Listo")
        self.progress_label.pack(side=tk.RIGHT)

        # Área de resultados en tiempo real
        results_frame = ttk.LabelFrame(sim_frame, text="📊 Resultados en Tiempo Real")
        results_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.results_text = tk.Text(results_frame, height=15, wrap=tk.WORD, font=("Consolas", 10))
        results_scrollbar = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=results_scrollbar.set)

        self.results_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        results_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    def setup_analysis_tab(self):
        """Configura la pestaña de análisis de resultados"""
        analysis_frame = ttk.Frame(self.notebook)
        self.notebook.add(analysis_frame, text="📈 Análisis")

        # Botones de análisis
        btn_frame = ttk.Frame(analysis_frame)
        btn_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Button(
            btn_frame,
            text="📊 Generar Reporte Completo",
            command=self.generar_reporte_completo
        ).pack(side=tk.LEFT, padx=5)

        ttk.Button(
            btn_frame,
            text="📈 Visualizar Gráficos",
            command=self.mostrar_graficos
        ).pack(side=tk.LEFT, padx=5)

        ttk.Button(
            btn_frame,
            text="💾 Guardar Reporte",
            command=self.guardar_reporte
        ).pack(side=tk.LEFT, padx=5)

        # Área de reporte
        report_frame = ttk.LabelFrame(analysis_frame, text="📋 Reporte Estadístico")
        report_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.report_text = tk.Text(report_frame, wrap=tk.WORD, font=("Consolas", 10))
        report_scrollbar = ttk.Scrollbar(report_frame, orient=tk.VERTICAL, command=self.report_text.yview)
        self.report_text.configure(yscrollcommand=report_scrollbar.set)

        self.report_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        report_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    # Métodos de la GUI original (adaptados)
    def toggle_auto_play(self):
        """Alternar entre juego automático y manual"""
        if self.auto_playing:
            self.stop_auto_play()
        else:
            self.start_auto_play()

    def start_auto_play(self):
        """Iniciar juego automático"""
        if not self.auto_playing and not self.game_over:
            strategy_name = self.strategy_var.get()
            if strategy_name == "Manual":
                messagebox.showwarning("Advertencia", "Selecciona una estrategia de IA primero")
                return

            self.auto_playing = True
            self.btn_auto.config(text="⏸ Pausar")

            # Crear estrategia
            if strategy_name == "Aleatoria":
                self.current_strategy = EstrategiaAleatoria()
            elif strategy_name == "Prioridad Derecha":
                self.current_strategy = EstrategiaPrioridadDerecha()
            elif strategy_name == "Optimal 2048":
                profundidad = int(self.depth_var.get()) if hasattr(self, 'depth_var') else 3
                self.current_strategy = EstrategiaOptimal2048(profundidad)

            self.play_next_move()

    def stop_auto_play(self):
        """Detener juego automático"""
        self.auto_playing = False
        self.btn_auto.config(text="▶ Jugar Automático")

    def play_next_move(self):
        """Ejecutar siguiente movimiento automático"""
        if not self.auto_playing or self.game_over:
            return

        move = self.current_strategy.obtener_movimiento(self.board)
        if move:
            moved = self.ejecutar_movimiento(move)
            if moved:
                self.add_random_tile()
                self.draw_board()
                self.update_info()

                if self.is_game_over():
                    self.game_over = True
                    self.show_game_over()
                    self.stop_auto_play()

        if self.auto_playing and not self.game_over:
            self.master.after(self.move_delay, self.play_next_move)

    def change_theme(self, theme_name):
        """Cambiar tema visual"""
        self.current_theme = THEMES[theme_name]
        self.canvas.config(bg=self.current_theme["background"])
        self.draw_board()

    def draw_board(self):
        """Dibujar el tablero"""
        self.canvas.delete("all")
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                value = self.board[i][j]
                x = j * self.cell_size + (j + 1) * self.padding
                y = i * self.cell_size + (i + 1) * self.padding

                self.canvas.create_rectangle(
                    x, y,
                    x + self.cell_size, y + self.cell_size,
                    fill=self.current_theme["cell_colors"][value],
                    outline=self.current_theme["background"],
                    width=0
                )

                if value != 0:
                    font_size = 40 if value < 100 else 30 if value < 1000 else 20
                    self.canvas.create_text(
                        x + self.cell_size // 2,
                        y + self.cell_size // 2,
                        text=str(value),
                        font=("Arial", font_size, "bold"),
                        fill=self.font_colors.get(value, "#f9f6f2")
                    )

    def key_press(self, event):
        """Manejo de teclas"""
        if self.auto_playing:
            return

        moved = False
        if event.keysym in ('Up', 'w', 'W'):
            moved = self.move_up()
        elif event.keysym in ('Down', 's', 'S'):
            moved = self.move_down()
        elif event.keysym in ('Left', 'a', 'A'):
            moved = self.move_left()
        elif event.keysym in ('Right', 'd', 'D'):
            moved = self.move_right()

        if moved:
            self.add_random_tile()
            self.draw_board()
            self.update_info()

            if self.is_game_over():
                self.show_game_over()

    def undo_move(self):
        """Deshacer último movimiento"""
        if self.history and not self.auto_playing:
            state = self.history.pop()
            self.board = state["board"]
            self.score = state["score"]
            self.max_tile = state["max_tile"]
            self.update_info()
            self.draw_board()

    def update_info(self):
        """Actualizar información del juego"""
        self.score_label.config(text=f"Puntuación: {self.score}")
        self.move_label.config(text=f"Movimientos: {self.move_count}")
        self.max_tile_label.config(text=f"Máxima ficha: {self.max_tile}")

        if self.score > self.high_score:
            self.high_score = self.score
            self.high_score_label.config(text=f"Récord: {self.high_score}")

    def show_game_over(self):
        """Mostrar mensaje de fin de juego"""
        self.canvas.create_rectangle(
            self.padding,
            self.padding + self.cell_size * self.grid_size // 2 - 60,
            self.grid_size * self.cell_size + self.padding * (self.grid_size + 1),
            self.padding + self.cell_size * self.grid_size // 2 + 60,
            fill="#edc22e",
            outline=""
        )
        self.canvas.create_text(
            self.grid_size * self.cell_size // 2 + self.padding * 2,
            self.padding + self.cell_size * self.grid_size // 2 - 10,
            text="¡Juego terminado!",
            font=("Arial", 32, "bold"),
            fill="#ffffff"
        )
        self.canvas.create_text(
            self.grid_size * self.cell_size // 2 + self.padding * 2,
            self.padding + self.cell_size * self.grid_size // 2 + 30,
            text=f"Puntuación: {self.score}",
            font=("Arial", 20, "bold"),
            fill="#ffffff"
        )

    def reset_game(self):
        """Reinicia el juego (específico para GUI)"""
        # Llama al reset del motor
        super().reset_game()

        # Actualiza la interfaz
        self.update_info()
        self.draw_board()

        # Reinicia historial y estado de juego
        self.history = []
        self.game_over = False

        # Detiene el juego automático si está activo
        if self.auto_playing:
            self.stop_auto_play()

    # Sobrescribir métodos de movimiento para guardar historial
    def move_up(self):
        self.history.append({
            "board": self.board.copy(),
            "score": self.score,
            "max_tile": self.max_tile
        })
        return super().move_up()

    def move_down(self):
        self.history.append({
            "board": self.board.copy(),
            "score": self.score,
            "max_tile": self.max_tile
        })
        return super().move_down()

    def move_left(self):
        self.history.append({
            "board": self.board.copy(),
            "score": self.score,
            "max_tile": self.max_tile
        })
        return super().move_left()

    def move_right(self):
        self.history.append({
            "board": self.board.copy(),
            "score": self.score,
            "max_tile": self.max_tile
        })
        return super().move_right()

    # Métodos de simulación Monte Carlo
    def iniciar_simulacion_threaded(self):
        """Iniciar simulación en hilo separado"""
        if self.experimento_ejecutandose:
            return

        # Validar configuración
        try:
            num_sims = int(self.sim_count_var.get())
            semilla = int(self.seed_var.get())
            profundidad = int(self.depth_var.get())
        except ValueError:
            messagebox.showerror("Error", "Valores de configuración inválidos")
            return

        # Verificar estrategias seleccionadas
        estrategias_seleccionadas = [name for name, var in self.strategy_vars.items() if var.get()]
        if not estrategias_seleccionadas:
            messagebox.showerror("Error", "Selecciona al menos una estrategia")
            return

        # Configurar experimento
        config = ExperimentConfig(
            num_simulaciones=num_sims,
            semilla_aleatoria=semilla,
            profundidad_expectiminimax=profundidad
        )

        self.simulador = SimuladorMonteCarlo(config)
        self.experimento_ejecutandose = True

        # Deshabilitar botón y habilitar detener
        self.btn_start_sim.config(state=tk.DISABLED)
        self.btn_stop_sim.config(state=tk.NORMAL)

        # Limpiar resultados
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, "🚀 Iniciando simulación Monte Carlo...\n\n")

        # Ejecutar en hilo separado
        self.simulation_thread = threading.Thread(target=self.ejecutar_simulacion, args=(estrategias_seleccionadas,))
        self.simulation_thread.daemon = True
        self.simulation_thread.start()

    def ejecutar_simulacion(self, estrategias_nombres):
        """Ejecutar simulación Monte Carlo"""
        try:
            # Crear instancias de estrategias
            estrategias = []
            profundidad = int(self.depth_var.get())

            for nombre in estrategias_nombres:
                if nombre == "Aleatoria":
                    estrategias.append(EstrategiaAleatoria())
                elif nombre == "Prioridad Derecha":
                    estrategias.append(EstrategiaPrioridadDerecha())
                elif nombre == "Optimal 2048":
                    estrategias.append(EstrategiaOptimal2048(profundidad))

            # Ejecutar experimento
            resultados = self.simulador.ejecutar_experimento(
                estrategias,
                callback_progreso=self.actualizar_progreso
            )

            # Al finalizar
            self.master.after(0, self.finalizar_simulacion)

        except Exception as e:
            self.master.after(0, lambda: self.error_simulacion(str(e)))

    def actualizar_progreso(self, porcentaje, estrategia_actual, simulacion_actual):
        """Actualizar progreso de simulación (thread-safe)"""

        def update():
            self.progress_var.set(porcentaje)
            self.progress_label.config(text=f"{estrategia_actual}: {simulacion_actual} - {porcentaje:.1f}%")

            # Añadir log cada 100 simulaciones
            if simulacion_actual % 100 == 0:
                total = self.simulador.config.num_simulaciones
                self.results_text.insert(tk.END, f"⏳ {estrategia_actual}: Simulación {simulacion_actual} de {total}\n")
                self.results_text.see(tk.END)

        if self.master:
            self.master.after(0, update)

    def finalizar_simulacion(self):
        """Finalizar simulación y mostrar resultados"""
        self.experimento_ejecutandose = False
        self.btn_start_sim.config(state=tk.NORMAL)
        self.btn_stop_sim.config(state=tk.DISABLED)
        self.progress_var.set(100)
        self.progress_label.config(text="✅ Simulación completada")
        self.results_text.insert(tk.END, "\n✅ Simulación completada exitosamente!\n")
        self.results_text.see(tk.END)

        messagebox.showinfo("Simulación Completa",
                            "✅ Simulación Monte Carlo completada exitosamente!\n\n"
                            "Ve a la pestaña 'Análisis' para ver el reporte completo.")

    def error_simulacion(self, error_msg):
        """Manejar error en simulación"""
        self.experimento_ejecutandose = False
        self.btn_start_sim.config(state=tk.NORMAL)
        self.btn_stop_sim.config(state=tk.DISABLED)
        self.progress_label.config(text="❌ Error en simulación")
        self.results_text.insert(tk.END, f"\n❌ Error: {error_msg}\n")
        self.results_text.see(tk.END)

        messagebox.showerror("Error en Simulación", f"Error durante la simulación:\n{error_msg}")

    def detener_simulacion(self):
        """Detener simulación en curso"""
        if self.simulador:
            self.simulador.detener_experimento()
        self.experimento_ejecutandose = False
        self.btn_start_sim.config(state=tk.NORMAL)
        self.btn_stop_sim.config(state=tk.DISABLED)
        self.progress_label.config(text="⏹ Simulación detenida")
        self.results_text.insert(tk.END, "\n⏹ Simulación detenida manualmente\n")
        self.results_text.see(tk.END)

    def exportar_resultados(self):
        """Exportar resultados a CSV"""
        if not self.simulador or not self.simulador.resultados:
            messagebox.showwarning("Sin Datos", "No hay resultados para exportar")
            return

        archivo = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            title="Guardar resultados"
        )

        if archivo:
            try:
                mensaje = self.simulador.exportar_datos_csv(archivo)
                messagebox.showinfo("Exportación Exitosa", mensaje)
            except Exception as e:
                messagebox.showerror("Error", f"Error al exportar: {str(e)}")

    def generar_reporte_completo(self):
        """Generar reporte estadístico completo"""
        if not self.simulador or not self.simulador.resultados:
            messagebox.showwarning("Sin Datos", "Ejecuta una simulación primero")
            return

        reporte = self.simulador.generar_reporte_completo()
        self.report_text.delete(1.0, tk.END)
        self.report_text.insert(tk.END, reporte)

    def mostrar_graficos(self):
        """Mostrar gráficos de análisis"""
        if not self.simulador or not self.simulador.resultados:
            messagebox.showwarning("Sin Datos", "Ejecuta una simulación primero")
            return

        try:
            self.crear_graficos_analisis()
        except ImportError:
            messagebox.showerror("Error", "Matplotlib no está instalado. Instala con: pip install matplotlib")
        except Exception as e:
            messagebox.showerror("Error", f"Error al generar gráficos: {str(e)}")

    def crear_graficos_analisis(self):
        """Crear gráficos de análisis estadístico"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Análisis Estadístico - Simulación Monte Carlo 2048', fontsize=16, fontweight='bold')

        # Preparar datos
        estrategias = list(self.simulador.resultados.keys())
        datos_puntuacion = []
        datos_movimientos = []
        datos_tiempo = []

        for estrategia in estrategias:
            resultados = self.simulador.resultados[estrategia]
            puntuaciones = [r.puntuacion for r in resultados]
            movimientos = [r.movimientos for r in resultados]
            tiempos = [r.tiempo_ejecucion for r in resultados]

            datos_puntuacion.append(puntuaciones)
            datos_movimientos.append(movimientos)
            datos_tiempo.append(tiempos)

        # Gráfico 1: Box plot de puntuaciones
        axes[0, 0].boxplot(datos_puntuacion, labels=estrategias)
        axes[0, 0].set_title('Distribución de Puntuaciones por Estrategia')
        axes[0, 0].set_ylabel('Puntuación')
        axes[0, 0].tick_params(axis='x', rotation=45)

        # Gráfico 2: Comparación de medias
        medias_puntuacion = [np.mean(datos) for datos in datos_puntuacion]
        bars = axes[0, 1].bar(estrategias, medias_puntuacion, color=['#ff9999', '#66b3ff', '#99ff99'])
        axes[0, 1].set_title('Puntuación Promedio por Estrategia')
        axes[0, 1].set_ylabel('Puntuación Promedio')
        axes[0, 1].tick_params(axis='x', rotation=45)

        # Añadir valores en las barras
        for bar, media in zip(bars, medias_puntuacion):
            height = bar.get_height()
            axes[0, 1].text(bar.get_x() + bar.get_width() / 2., height,
                            f'{media:.0f}', ha='center', va='bottom')

        # Gráfico 3: Comparación de movimientos
        medias_movimientos = [np.mean(datos) for datos in datos_movimientos]
        bars2 = axes[1, 0].bar(estrategias, medias_movimientos, color=['#ffcc99', '#ff99cc', '#ccff99'])
        axes[1, 0].set_title('Promedio de Movimientos por Partida')
        axes[1, 0].set_ylabel('Movimientos')
        axes[1, 0].tick_params(axis='x', rotation=45)

        # Añadir valores en las barras
        for bar, media in zip(bars2, medias_movimientos):
            height = bar.get_height()
            axes[1, 0].text(bar.get_x() + bar.get_width() / 2., height,
                            f'{media:.0f}', ha='center', va='bottom')

        # Gráfico 4: Eficiencia (Puntos por segundo)
        medias_tiempo = [np.mean(datos) for datos in datos_tiempo]
        eficiencia = [p / t for p, t in zip(medias_puntuacion, medias_tiempo)]
        bars3 = axes[1, 1].bar(estrategias, eficiencia, color=['#ffb3ba', '#baffc9', '#bae1ff'])
        axes[1, 1].set_title('Eficiencia (Puntos por Segundo)')
        axes[1, 1].set_ylabel('Puntos/Segundo')
        axes[1, 1].tick_params(axis='x', rotation=45)

        # Añadir valores en las barras
        for bar, ef in zip(bars3, eficiencia):
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width() / 2., height,
                            f'{ef:.0f}', ha='center', va='bottom')

        plt.tight_layout()
        plt.show()

    def guardar_reporte(self):
        """Guardar reporte en archivo"""
        if not self.simulador or not self.simulador.resultados:
            messagebox.showwarning("Sin Datos", "No hay reporte para guardar")
            return

        archivo = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            title="Guardar reporte"
        )

        if archivo:
            try:
                reporte = self.simulador.generar_reporte_completo()
                with open(archivo, 'w', encoding='utf-8') as f:
                    f.write(reporte)
                messagebox.showinfo("Guardado Exitoso", f"Reporte guardado en: {archivo}")
            except Exception as e:
                messagebox.showerror("Error", f"Error al guardar: {str(e)}")


# Método para crear rectángulos redondeados (función auxiliar)
def create_round_rect(self, x1, y1, x2, y2, radius=5, **kwargs):
    return self.create_polygon(
        x1 + radius, y1,
        x2 - radius, y1,
        x2, y1 + radius,
        x2, y2 - radius,
        x2 - radius, y2,
        x1 + radius, y2,
        x1, y2 - radius,
        x1, y1 + radius,
        x1 + radius, y1,
        smooth=True,
        **kwargs
    )


tk.Canvas.create_round_rect = create_round_rect


# Función principal para ejecutar simulación desde línea de comandos
def ejecutar_simulacion_cli():
    """Ejecutar simulación desde línea de comandos sin GUI"""
    print("🎮 Iniciando simulación Monte Carlo 2048 (modo CLI)")
    print("=" * 60)

    # Configuración
    config = ExperimentConfig(
        num_simulaciones=1000,
        semilla_aleatoria=42,
        profundidad_expectiminimax=3
    )

    # Crear estrategias
    estrategias = [
        EstrategiaAleatoria(),
        EstrategiaPrioridadDerecha(),
        EstrategiaOptimal2048(config.profundidad_expectiminimax)
    ]

    # Ejecutar simulación
    simulador = SimuladorMonteCarlo(config)
    resultados = simulador.ejecutar_experimento(estrategias)

    # Mostrar reporte
    print("\n" + "=" * 60)
    print("🎉 SIMULACIÓN COMPLETADA")
    print("=" * 60)

    reporte = simulador.generar_reporte_completo()
    print(reporte)

    # Exportar resultados
    archivo_csv = f"resultados_2048_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    mensaje_export = simulador.exportar_datos_csv(archivo_csv)
    print(f"\n{mensaje_export}")

    return simulador


# Función de ejemplo para pruebas rápidas
def test_estrategias():
    """Función de prueba rápida para las estrategias"""
    print("🧪 Probando estrategias...")

    # Crear juego de prueba
    juego = Game2048Engine()
    juego.reset_game()

    # Estado de prueba
    juego.board = np.array([
        [2, 4, 8, 16],
        [4, 8, 16, 32],
        [8, 16, 32, 64],
        [0, 0, 0, 0]
    ])

    print(f"Estado inicial del tablero:")
    print(juego.board)
    print()

    # Probar cada estrategia
    estrategias = [
        EstrategiaAleatoria(),
        EstrategiaPrioridadDerecha(),
        EstrategiaOptimal2048(3)
    ]

    for estrategia in estrategias:
        movimiento = estrategia.obtener_movimiento(juego.board.copy())
        print(f"{estrategia.nombre}: {movimiento}")

    print("\n✅ Prueba de estrategias completada")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "cli":
        # Ejecutar en modo CLI
        simulador = ejecutar_simulacion_cli()
    elif len(sys.argv) > 1 and sys.argv[1] == "test":
        # Ejecutar pruebas
        test_estrategias()
    else:
        # Ejecutar con GUI
        try:
            root = tk.Tk()
            root.geometry("1000x700")
            app = Game2048(root)


            # Configurar para cerrar correctamente
            def on_closing():
                if hasattr(app, 'experimento_ejecutandose') and app.experimento_ejecutandose:
                    if messagebox.askokcancel("Salir", "¿Detener simulación en curso y salir?"):
                        if hasattr(app, 'simulador') and app.simulador:
                            app.simulador.detener_experimento()
                        root.destroy()
                else:
                    root.destroy()


            root.protocol("WM_DELETE_WINDOW", on_closing)
            root.mainloop()

        except KeyboardInterrupt:
            print("\n🛑 Ejecución interrumpida por el usuario")
        except Exception as e:
            print(f"❌ Error al ejecutar la aplicación: {e}")
            import traceback

            traceback.print_exc()