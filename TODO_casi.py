#Hola mi bida
# Este es un intento :p 
#no se que hago pero luzco genial haciendolo
#ehm pues creo que aqui se calcula el área, por procesamiento de imagenes y por montecarlo
#se supone que también esta la comparación de amnos métodos pero no se por que se ve asi la gráfica
#segun también esta la Estimación de análisis en tiempo y frecuencia de señales acústicas sobre muestra
#tambien se estima el área de adhesión :p efectiva
#lo que si me confundío es el mapa de posición y amplitud del minimo, creo que ese esta mal, debería ser lo que estabamos haciendo ayer 

#muchas gracias por ayudarme mi amor, te amo c: muchote

import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.stats import linregress
np.random.seed(123)  # Reproducibilidad

# --- PARÁMETROS GLOBALES (ACTUALIZADO) ---
IMAGEN_PATH = 'ima_rec.JPG'  # Tu imagen
CSV_PATH = r'C:\Users\qbo28\OneDrive\Escritorio\Proyecto_mediotermino\sixty_signal_adhesive.csv'  # Tu CSV
scale_px_per_cm = 1565 / 15  # Nueva escala: 104.333 px/cm
k_muestras = 60
error_deseado = 0.05
sigma_adhesion = 0.2
f_lim = 0.1e7  # 100 MHz
num_simulaciones = 200
N_PUNTOS_MC = 100000  # Para área MC

print("=== PROYECTO END COMPLETO: Cobertura de 6 Puntos (Escala Actualizada) ===")

# 1. SEGMENTACIÓN (Punto 1: Área por procesamiento de imágenes)
print("\n1. Estimación Área por Imágenes...")
imagen = cv2.imread(IMAGEN_PATH, cv2.IMREAD_GRAYSCALE)
if imagen is None: raise ValueError(f"Error cargando {IMAGEN_PATH}")
imagen_suavizada = cv2.GaussianBlur(imagen, (5, 5), 0)
umbral, binarizada = cv2.threshold(imagen_suavizada, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
binarizada = cv2.bitwise_not(binarizada)
kernel = np.ones((3, 3), np.uint8)
binarizada_limpia = cv2.morphologyEx(binarizada, cv2.MORPH_CLOSE, kernel, iterations=1)
binarizada_limpia = cv2.morphologyEx(binarizada_limpia, cv2.MORPH_OPEN, kernel, iterations=1)
labels, num_labels = ndimage.label(binarizada_limpia > 0)
sizes = ndimage.sum(binarizada_limpia > 0, labels, range(num_labels + 1))
largest_label = sizes.argmax()
adhesivo_mask = (labels == largest_label)
area_px = np.sum(adhesivo_mask)
area_cm2 = area_px / (scale_px_per_cm ** 2)
print(f"Área por imágenes: {area_cm2:.2f} cm²")

# Gráfico Punto 1
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1); plt.imshow(imagen, cmap='gray'); plt.title('Original'); plt.axis('off')
plt.subplot(1, 3, 2); plt.imshow(binarizada_limpia, cmap='gray'); plt.title('Binarizada'); plt.axis('off')
plt.subplot(1, 3, 3); plt.imshow(adhesivo_mask, cmap='gray'); plt.title('Máscara'); plt.axis('off')
plt.tight_layout()
plt.savefig('segmentacion.png', dpi=150, bbox_inches='tight')
plt.show()

# 2. MONTE CARLO PARA ÁREA (Punto 2)
print("\n2. Estimación Área por Monte Carlo...")
h, w = imagen.shape
puntos_x = np.random.uniform(0, w, N_PUNTOS_MC)
puntos_y = np.random.uniform(0, h, N_PUNTOS_MC)
coords_x = np.clip(np.round(puntos_x).astype(int), 0, w-1)
coords_y = np.clip(np.round(puntos_y).astype(int), 0, h-1)
resultados = adhesivo_mask[coords_y, coords_x]
puntos_dentro = np.sum(resultados)
area_total_muestreo = w * h
area_estimada_mc = area_total_muestreo * (puntos_dentro / N_PUNTOS_MC)
area_mc_cm2 = area_estimada_mc / (scale_px_per_cm ** 2)
print(f"Área por MC: {area_mc_cm2:.2f} cm²")

# Gráfico Punto 2
puntos_mc = np.column_stack((puntos_x, puntos_y))
puntos_internos = puntos_mc[resultados]
imagen_rgb = cv2.cvtColor(cv2.imread(IMAGEN_PATH), cv2.COLOR_BGR2RGB)
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1); plt.imshow(imagen_rgb); plt.title('Original'); plt.axis('off')
plt.subplot(1, 2, 2); plt.imshow(imagen_rgb); plt.scatter(puntos_internos[:, 0], puntos_internos[:, 1], s=0.5, c='green', alpha=0.5)
plt.title(f'MC: {area_mc_cm2:.2f} cm²'); plt.axis('off')
plt.tight_layout()
plt.savefig('montecarlo_area.png', dpi=150, bbox_inches='tight')
plt.show()

# 3. COMPARACIÓN Y ERROR VS. K (Punto 3)
print("\n3. Comparación y Error vs. k...")
error_relativo = abs(area_mc_cm2 - area_cm2) / area_cm2 * 100
print(f"Comparación: Error relativo {error_relativo:.1f}%")
k_range = np.arange(1, 101)
se_directo = np.zeros_like(k_range)  # Directo: converge inmediatamente
se_mc = sigma_adhesion / np.sqrt(k_range)  # MC: SE teórico

plt.figure(figsize=(8, 5))
plt.plot(k_range, se_directo, 'b-', label='Directo (Otsu)')
plt.plot(k_range, se_mc, 'r--', label='Monte Carlo')
plt.axhline(error_deseado, color='g', linestyle=':', label=f'Error Deseado ({error_deseado})')
plt.xlabel('Número de Tomas (k)'); plt.ylabel('Error Estándar (SE)'); plt.title('Error vs. k')
plt.legend(); plt.grid(alpha=0.3)
plt.savefig('error_vs_k.png', dpi=150, bbox_inches='tight')
plt.show()

# 4. ANÁLISIS TIEMPO/FRECUENCIA (Punto 4)
print("\n4. Análisis Tiempo/Frecuencia...")
data = pd.read_csv(CSV_PATH)
t = data.iloc[:, 0].values
signals = data.iloc[:, 1:].values
num_signals_total = signals.shape[1]
dt = np.diff(t); dt = dt[dt > 0]; fs = 1 / np.mean(dt)
print(f"fs: {fs/1e6:.2f} MHz")

num_signals_viz = min(10, num_signals_total)
signals_10 = signals[:, :num_signals_viz]
N = len(t)
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
for i in range(num_signals_viz): plt.plot(t, signals_10[:, i], label=f'S{i+1}', alpha=0.7)
plt.xlabel('Tiempo (s)'); plt.ylabel('Amplitud'); plt.title('Tiempo (Primeras 10)'); plt.legend(ncol=2); plt.grid(alpha=0.3)

plt.subplot(1, 2, 2)
Y_10 = np.fft.fft(signals_10, axis=0)
f = np.linspace(0, fs / 2, N // 2)
mask_f = f <= f_lim
for i in range(num_signals_viz):
    mag = np.abs(Y_10[:N // 2, i][mask_f])
    plt.plot(f[mask_f]/1e6, mag, label=f'S{i+1}', alpha=0.7)
plt.xlabel('Frecuencia (MHz)'); plt.ylabel('Magnitud'); plt.title(f'Frecuencia (hasta {f_lim/1e6:.1f} MHz)')
plt.legend(ncol=2); plt.grid(alpha=0.3); plt.xscale('log')
plt.tight_layout()
plt.savefig('senales_tiempo_frecuencia.png', dpi=150, bbox_inches='tight')  # Una figura combinada
plt.show()

# Cálculo f0
f0_values = []
Y_total = np.fft.fft(signals, axis=0)
magnitudes = np.abs(Y_total[:N // 2, :][mask_f, :])
for i in range(num_signals_total):
    mag_i = magnitudes[:, i]
    peak_idx = np.argmax(mag_i[1:]) + 1
    f0_i = f[mask_f][peak_idx]
    f0_values.append(f0_i)
f0_mhz = np.array(f0_values) / 1e6
print(f"Media f0: {np.mean(f0_mhz):.3f} MHz")

# 5. COMBINACIÓN MC + ESPECTRAL (Punto 5: Área de Adhesión)
print("\n5. Combinación MC + Espectral...")
y_coords, x_coords = np.where(adhesivo_mask)
num_puntos_validos = len(x_coords)
indices_muestreo = np.random.choice(num_puntos_validos, k_muestras, replace=False)
posiciones_px = np.column_stack((x_coords[indices_muestreo], y_coords[indices_muestreo]))
posiciones_cm = posiciones_px / scale_px_per_cm
pos_mapeadas_cm = posiciones_cm[:num_signals_total]
adhesion_efectiva = area_cm2 * np.mean(f0_mhz)  # Integración: área × media f0
print(f"Adhesión efectiva (área adhesión ponderada): {adhesion_efectiva:.3f} MHz·cm²")

# 6. MAPA DEL "MÍNIMO" (Punto 6: Dos gráficos)
print("\n6. Mapa del Mínimo...")
amplitud_minima = []
for i in range(num_signals_total):
    mag_i = magnitudes[:, i]
    peak_idx = np.argmax(mag_i[1:]) + 1
    valle_range = slice(peak_idx, min(peak_idx + 20, len(mag_i)))
    valle_idx = peak_idx + np.argmin(mag_i[valle_range] - mag_i[peak_idx])  # Valle relativo post-pico
    amp_min_i = mag_i[valle_idx]
    amplitud_minima.append(amp_min_i)
amp_min = np.array(amplitud_minima)

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.imshow(adhesivo_mask, cmap='gray', extent=[0, w/scale_px_per_cm, 0, h/scale_px_per_cm])
plt.scatter(pos_mapeadas_cm[:, 0], pos_mapeadas_cm[:, 1], c=amp_min, s=50, cmap='coolwarm', edgecolors='k')
plt.colorbar(label='Amplitud Mínimo'); plt.title('Posición y Amplitud del Mínimo')
plt.xlabel('X (cm)'); plt.ylabel('Y (cm)'); plt.axis('equal')

plt.subplot(1, 2, 2)
plt.hist2d(pos_mapeadas_cm[:, 0], pos_mapeadas_cm[:, 1], bins=10, weights=amp_min, cmap='coolwarm')
plt.colorbar(label='Amplitud Mínimo'); plt.title('Distribución Espacial del Mínimo')
plt.xlabel('X (cm)'); plt.ylabel('Y (cm)')
plt.tight_layout()
plt.savefig('mapa_minimo.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"Media amplitud mínimo: {np.mean(amp_min):.2e}")

# Exportaciones
df_pos = pd.DataFrame({'x_cm': pos_mapeadas_cm[:, 0], 'y_cm': pos_mapeadas_cm[:, 1]})
df_pos.to_csv('posiciones_muestreo.csv', index=False)
df_f0 = pd.DataFrame({'D': range(1, num_signals_total+1), 'f0_MHz': f0_mhz, 'amp_min': amp_min})
df_f0.to_csv('f0_y_minimo.csv', index=False)
print("\n=== Figuras y CSVs generados para reporte. Todo cubierto! ===")