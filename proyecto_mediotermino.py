#ESTE SIII
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
from scipy import ndimage  

np.random.seed(123)

# --- PARÁMETROS ---
N_PUNTOS = 100000  # Número de puntos para la simulación de Montecarlo (área)
IMAGEN_PATH = 'ima_rec.JPG'  # Ajusta al nombre de tu archivo (usa .JPG según el nuevo código)

# Parámetros para estimación de k mínimo
scale_px_per_cm = 1565 / 15  # Nueva escala: 104.333 px/cm
scale_px_per_cm2 = scale_px_per_cm * scale_px_per_cm
num_simulaciones = 1000  # Número de simulaciones Monte Carlo para SE
max_muestras = 400  # Máximo k a probar
error_deseado = 0.05  # Error deseado para SE
sigma_adhesion = 0.2  # Desviación simulada de adhesión

#%% 1. PRE-PROCESAMIENTO CON OTSU (nuevo método proporcionado)
# Cargar la imagen en escala de grises
imagen = cv2.imread(IMAGEN_PATH, cv2.IMREAD_GRAYSCALE)
if imagen is None:
    print(f"Error: No se pudo cargar la imagen en {IMAGEN_PATH}")
    exit()

# Aplicar un suavizado (opcional pero recomendable)
imagen_suavizada = cv2.GaussianBlur(imagen, (5, 5), 0)

# Aplicar el método de Otsu
umbral, binarizada = cv2.threshold(imagen_suavizada, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Invertir si es necesario (asumiendo que el adhesivo es el fondo oscuro; ajusta según tu imagen)
# Si el objeto es blanco después de Otsu, comenta la siguiente línea
binarizada = cv2.bitwise_not(binarizada)

print(f"Umbral calculado por Otsu: {umbral:.2f}")

# Limpieza opcional: operaciones morfológicas para remover ruido
kernel = np.ones((3, 3), np.uint8)
binarizada_limpia = cv2.morphologyEx(binarizada, cv2.MORPH_CLOSE, kernel, iterations=1)
binarizada_limpia = cv2.morphologyEx(binarizada_limpia, cv2.MORPH_OPEN, kernel, iterations=1)

# Etiquetado de regiones conectadas para seleccionar la principal
labels, num_labels = ndimage.label(binarizada_limpia > 0)
if num_labels == 0:
    print("Error: No se encontraron regiones conectadas.")
    exit()

# Seleccionar la región más grande (adhesivo principal)
sizes = ndimage.sum(binarizada_limpia > 0, labels, range(num_labels + 1))
largest_label = sizes.argmax()
adhesivo_mask = (labels == largest_label)

# Área directa (para comparación)
area_px_directa = np.sum(adhesivo_mask)
print(f'Área directa en píxeles: {area_px_directa}')

# Escala física (opcional: ajusta según tu imagen)
scale_px_per_cm = 106.0  # Ejemplo; mide la regla en píxeles
area_cm2_directa = area_px_directa / (scale_px_per_cm ** 2)
print(f'Área directa estimada en cm²: {area_cm2_directa:.2f}')

# Visualización del pre-procesamiento (del nuevo código)
plt.figure(figsize=(12, 5))
plt.subplot(1, 3, 1)
plt.title('Imagen original')
plt.imshow(imagen, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title('Imagen suavizada')
plt.imshow(imagen_suavizada, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title('Binarizada (Otsu)')
plt.imshow(binarizada, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()

#%% 2. APLICACIÓN DEL MÉTODO DE MONTECARLO PARA ÁREA
# Definir el Área de Muestreo (Bounding Box)
h, w = imagen.shape
x_min, x_max = 0, w
y_min, y_max = 0, h

# Generar Puntos Aleatorios
puntos_x = np.random.uniform(x_min, x_max, N_PUNTOS)
puntos_y = np.random.uniform(y_min, y_max, N_PUNTOS)
puntos_mc = np.column_stack((puntos_x, puntos_y))

# Prueba de Inclusión usando la MÁSCARA
# Redondear coordenadas a enteros para indexar la máscara
coords_x = np.clip(np.round(puntos_x).astype(int), 0, w-1)
coords_y = np.clip(np.round(puntos_y).astype(int), 0, h-1)
resultados = adhesivo_mask[coords_y, coords_x]  # True si dentro

puntos_dentro = np.sum(resultados)
puntos_fuera = N_PUNTOS - puntos_dentro

# Estimación de área
area_total_muestreo = (x_max - x_min) * (y_max - y_min)
area_estimada_roi = area_total_muestreo * (puntos_dentro / N_PUNTOS)

# Preparar datos para visualización
puntos_internos = puntos_mc[resultados]
puntos_externos = puntos_mc[~resultados]

# Cargar imagen original en color para visualización
imagen_original = cv2.imread(IMAGEN_PATH)
if imagen_original is None:
    print(f"Error: No se pudo cargar la imagen en color en {IMAGEN_PATH}")
    exit()
# Convertir BGR a RGB para plt
imagen_rgb = cv2.cvtColor(imagen_original, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(12, 8))

# Subplot 1: Imagen original con máscara superpuesta
plt.subplot(2, 2, 1)
plt.imshow(imagen_rgb)
plt.imshow(adhesivo_mask, cmap='Reds', alpha=0.5)  # Superponer máscara roja semi-transparente
plt.title('Imagen Original con ROI (Máscara)')
plt.axis('off')

# Subplot 2: Binarizada limpia
plt.subplot(2, 2, 2)
plt.imshow(binarizada_limpia, cmap='gray')
plt.title('Binarizada Limpia + Región Principal')
plt.imshow(adhesivo_mask, cmap='YlOrRd', alpha=0.3)  # Superponer región en amarillo-naranja-rojo
plt.axis('off')

# Subplot 3: Monte Carlo con puntos
plt.subplot(2, 2, 3)
plt.imshow(imagen_rgb)
plt.title(f'Monte Carlo - ROI Estimada: {area_estimada_roi/scale_px_per_cm2:.2f} cm²\n(Directa: {area_px_directa/scale_px_per_cm2:.2f} cm²)')
plt.scatter(puntos_internos[:, 0], puntos_internos[:, 1], color='green', s=2, label='Dentro', alpha=0.6)
plt.scatter(puntos_externos[:, 0], puntos_externos[:, 1], color='red', s=2, alpha=0.1, label='Fuera')
plt.xlim(x_min, x_max)
plt.ylim(y_max, y_min)  # Invertir eje Y
plt.legend()

# Subplot 4: Solo la región seleccionada
plt.subplot(2, 2, 4)
plt.imshow(adhesivo_mask, cmap='gray')
plt.title('Región Adhesivo Seleccionada')
plt.axis('off')

plt.tight_layout()
plt.show()

print(f"\n--- Resultados Montecarlo (Área) ---")
print(f"Puntos Totales Generados: {N_PUNTOS}")
print(f"Puntos Dentro de la ROI: {puntos_dentro}")
print(f"Proporción (Puntos Dentro / Totales): {puntos_dentro / N_PUNTOS:.4f}")
print(f"Área Total de Muestreo (Bounding Box): {area_total_muestreo/scale_px_per_cm2} cm²")
print(f"Área Estimada de la ROI: {area_estimada_roi/scale_px_per_cm2:.2f} cm²")
print(f"Área Directa (para comparación): {area_px_directa/scale_px_per_cm2} cm²")

#%% 3. ESTIMACIÓN DEL K MÍNIMO (Monte Carlo para tamaño de muestra en adhesión)
# Coordenadas válidas en la máscara (puntos/píxeles del adhesivo)
y_coords, x_coords = np.where(adhesivo_mask)
num_puntos_validos = len(x_coords)
print(f'\nÁrea total del adhesivo: {num_puntos_validos} píxeles (~{num_puntos_validos / (scale_px_per_cm**2):.2f} cm²)')

# Simular valores de adhesión (normal: media=1.0, sigma=0.2) para cada píxel válido
valores_adhesion = np.random.normal(1.0, sigma_adhesion, num_puntos_validos)

# Monte Carlo para estimar n (error estándar de la media < error_deseado)
resultados_se = np.zeros(max_muestras)
for k in range(1, max_muestras + 1):
    medias_sim = []
    for _ in range(num_simulaciones):
        indices = np.random.choice(num_puntos_validos, k, replace=True)
        media_k = np.mean(valores_adhesion[indices])
        medias_sim.append(media_k)
    se_k = np.std(medias_sim) / np.sqrt(k)  # Aprox. SE de la media
    resultados_se[k-1] = se_k

# Encontrar k mínimo
k_minimo = np.argmax(resultados_se < error_deseado) + 1
if k_minimo == 0:
    k_minimo = max_muestras
print(f'Número mínimo de señales (muestras): {k_minimo}')
print(f'SE a {k_minimo} muestras: {resultados_se[k_minimo-1]:.3f} < {error_deseado}')

# Gráfica de convergencia
plt.figure(figsize=(8, 5))
plt.plot(resultados_se, label='Error Estándar (SE)')
plt.axhline(error_deseado, color='r', linestyle='--', label='Error Deseado')
plt.axvline(k_minimo, color='g', linestyle=':', label=f'k Mínimo: {k_minimo}')
plt.xlabel('Número de Muestras (k)')
plt.ylabel('SE de la Media Adhesión')
plt.title('Convergencia Monte Carlo para Tamaño de Muestra')
plt.legend()
plt.grid(True)
plt.show()

#%% 3. ESTIMACIÓN DEL K MÍNIMO (Monte Carlo para tamaño de muestra en adhesión)
# Coordenadas válidas en la máscara (puntos/píxeles del adhesivo)
y_coords, x_coords = np.where(adhesivo_mask)
num_puntos_validos = len(x_coords)
print(f'\nÁrea total del adhesivo: {num_puntos_validos} píxeles (~{num_puntos_validos / (scale_px_per_cm**2):.2f} cm²)')

# Parámetros (ya definidos arriba)
z = 1.96  # Para 95% confianza (ajusta si cambias confianza)

# Opción 1: Teórica (rápida y exacta para distribución normal)
resultados_se_teorico = sigma_adhesion / np.sqrt(np.arange(1, max_muestras + 1))
# Ajuste por CI: margen = z * SE
margenes_ci = z * resultados_se_teorico
k_minimo_teorico = np.argmax(margenes_ci < error_deseado) + 1
if k_minimo_teorico == 0:
    k_minimo_teorico = max_muestras
print(f'k mínimo teórico (margen CI 95% < {error_deseado}): {k_minimo_teorico}')
print(f'Margen a {k_minimo_teorico} muestras: {margenes_ci[k_minimo_teorico-1]:.3f} < {error_deseado}')

# Opción 2: Simulación Monte Carlo (para verificar; usa menos simulaciones para velocidad)
# Simular valores de adhesión (solo una vez, reproducible)
valores_adhesion = np.random.normal(1.0, sigma_adhesion, num_puntos_validos)
resultados_se_sim = np.zeros(max_muestras)
num_simulaciones_lite = 200  # Reducido para no tardar (aumenta si quieres precisión)
for k in range(1, max_muestras + 1):
    medias_sim = []
    for _ in range(num_simulaciones_lite):
        indices = np.random.choice(num_puntos_validos, k, replace=True)
        media_k = np.mean(valores_adhesion[indices])
        medias_sim.append(media_k)
    se_k = np.std(medias_sim)  # ¡CORREGIDO! Sin / sqrt(k)
    resultados_se_sim[k-1] = se_k

# Ajuste por CI
margenes_ci_sim = z * resultados_se_sim
k_minimo_sim = np.argmax(margenes_ci_sim < error_deseado) + 1
if k_minimo_sim == 0:
    k_minimo_sim = max_muestras
print(f'k mínimo simulado (margen CI 95% < {error_deseado}): {k_minimo_sim}')
print(f'Margen simulado a {k_minimo_sim} muestras: {margenes_ci_sim[k_minimo_sim-1]:.3f} < {error_deseado}')

# Gráfica de convergencia (usa teórica para plot)
plt.figure(figsize=(10, 6))
k_range = np.arange(1, max_muestras + 1)
plt.plot(k_range, resultados_se_teorico, label='SE Teórico (σ/√k)', color='blue')
plt.plot(k_range, resultados_se_sim[:len(k_range)], '--', label='SE Simulado', color='orange', alpha=0.7)
plt.axhline(error_deseado / z, color='r', linestyle='--', label=f'SE para margen {error_deseado} (e/z)')
plt.axvline(k_minimo_teorico, color='g', linestyle=':', label=f'k Mínimo Teórico: {k_minimo_teorico}')
plt.xlabel('Número de Muestras (k)')
plt.ylabel('Error Estándar (SE)')
plt.title('Convergencia Monte Carlo para Tamaño de Muestra (95% CI)')
plt.legend()
plt.grid(True)
plt.show()