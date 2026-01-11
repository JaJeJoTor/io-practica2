import matplotlib.pyplot as plt
import cv2
import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import cdist

class Morphing():
    """
    Entradas y salidas:
    Las entradas serán dos imágenes (img1 e img2) en formato jpeg. 
    La resolución de las imágenes no importa, ya que la redimensionamos más adelante en el preprocesado.
    """
    
    def __init__(self, img1_path, img2_path, resolucion = 168, epsilon=0.01, n_iter=1000, num_frames=10, color= True):
        self.img1 = cv2.imread(img1_path)
        self.img2 = cv2.imread(img2_path)

        if not color:
            self.img1 = cv2.cvtColor(self.img1, cv2.COLOR_BGR2GRAY)
            self.img2 = cv2.cvtColor(self.img2, cv2.COLOR_BGR2GRAY)

        self.shape = (resolucion, resolucion)
        self.epsilon = epsilon
        self.color = color


    def processing(self, img):
        '''Procesa una imagen para poder aplicar el algoritmo de sinkhorn.'''

        # Reducimos la resolución de las imágenes porque el algoritmo de 
        # sinkhorn escala mal con muchos píxeles. 
        # Por ejemplo, con imágenes de 512x512 píxeles,
        # la matriz de costes tendría 512*512 x 512*512 = 68 mil millones de entradas, 
        # en la RAM ocuparía demasiado espacio.
        img = cv2.resize(img, self.shape)
        
        # Convertimos la imagen a float, porque el algoritmo de sinkhorn
        # trata las imagenes como distribuciones de probabilidad. 
        # Al hacer las operaciones, necesitamos mucha precisión para dividir y multiplicar.
        img = img.astype('float32')
        
        # Aplanamos la imagen para tener un vector en lugar de una matriz 2D.
        # El algoritmo de sinkhorn no necesita estructura 2D, ya que no utiliza 
        # información de "arriba" o "abajo", sino que usa las imágenes como una lista
        # de cantidades de masa en cada píxel.
        img = img.flatten()

        # Dividimos entre la suma total de la imagen para convertirla 
        # en una distribución de probabilidad según la intensidad de los píxeles. 
        # También para que la suma total de masa en ambas imágenes sea la misma,
        # ya que la toda masa de un píxel de la imagen 1 se tiene que transportar a 
        # distintos píxeles o un pixel de la imagen 2 y viceversa, no se puede perder
        # nada de masa. 

        # Sumamos un valor muy pequeño para evitar ceros
        img += 1e-9
        img = img / img.sum()

        return img
    

    def make_costs_matrix(self):
        '''Realiza la matriz de costes dadas las dimensiones de una imagen.'''
        
        # Sin la matriz de coste, el algoritmo no sabe cómo de lejos están dos píxeles.
        # No sabría si el pixel (0,0) está cerca del píxel (0,1) o lejos del píxel (255,255).
        x = np.arange(self.shape[0])
        X, Y = np.meshgrid(x, x)
        coordenadas = np.stack([Y.ravel(), X.ravel()], axis=1) # (N^2, 2)
        
        # cdist calcula la distancia entre todos los pares de puntos, la disnancia que 
        # utilizamos es la euclídea al cuadrado.
        costs_matrix = cdist(coordenadas, coordenadas, metric='sqeuclidean')
        
        # Normalizamos la matriz de coste entre 0 y 1
        costs_matrix = costs_matrix / costs_matrix.max()

        return costs_matrix, coordenadas
    

    def get_kernel(self, costs_matrix):
        '''Construye el kernel de Sinkhorn.'''
        
        # Este paso transforma el problema de costes en uno de multiplicaciones
        K = np.exp(-costs_matrix / self.epsilon)
        
        return K
            
    
    def sinkhorn(self, img1, img2, K, n_iter = 100):
        '''Implementa el algoritmo de Sinkhorn para obtener la matriz de transporte óptimo.'''
        
        # Procesamos las imágenes para obtener las distribuciones de probabilidad.
        a = self.processing(img1)
        b = self.processing(img2)
        
        u = np.ones(len(a))
        v = np.ones(len(b))

        for i in tqdm(range(n_iter)):
            u = a / np.dot(K, v)
            v = b / np.dot(K.T, u)
        
        # En lugar de: P = np.diag(u) @ K @ np.diag(v)
        # Usamos broadcasting que es O(n²) en vez de O(n³)
        P = u[:, np.newaxis] * K * v[np.newaxis, :]

        return P
    
                
    def make_video(self, P, coordenadas, num_frames=60):
        '''Crea el video de transición del morphing entre las dos imágenes (versión vectorizada).'''
        
        frames = []
        t_values = np.linspace(0, 1, num_frames)
        
        # Encontrar todos los pares (i, j) donde hay transporte significativo
        # Esto reduce muchísimo el número de operaciones
        threshold = 1e-10
        i_indices, j_indices = np.where(P > threshold)
        masas = P[i_indices, j_indices]  # Masas transportadas
        
        # Coordenadas origen y destino para cada par con transporte
        coords_origen = coordenadas[i_indices]   # (num_pares, 2)
        coords_destino = coordenadas[j_indices]  # (num_pares, 2)
        
        for t in tqdm(t_values, desc="Generando frames"):
            
            # Interpolación de McCann vectorizada: posiciones intermedias
            pos_intermedias = (1 - t) * coords_origen + t * coords_destino  # (num_pares, 2)
            
            # Redondear a píxeles
            py = np.clip(np.round(pos_intermedias[:, 0]).astype(int), 0, self.shape[0] - 1)
            px = np.clip(np.round(pos_intermedias[:, 1]).astype(int), 0, self.shape[1] - 1)
            
            # Índices lineales
            indices = py * self.shape[1] + px
            
            # Acumular masas usando np.bincount (muy rápido)
            frame = np.bincount(indices, weights=masas, minlength=self.shape[0] * self.shape[1])
            
            # Normalizar para visualización
            frame = (frame / frame.max() * 255 + 1e-9).astype(np.uint8)
            frame = frame.reshape(self.shape)
            frames.append(frame)
        
        return frames

    def save_video(self, frames, output_path='morphing.mp4', fps=30):
        '''Guarda los frames como un video.'''
        
        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=self.color)
        
        for frame in frames:
            out.write(frame)
        
        out.release()
        print(f"Video guardado en {output_path}")
    

    def run(self, img1, img2, n_iter=100, num_frames=60):
        '''Ejecuta todo el pipeline de morphing.'''
        
        print("Creando matriz de costes...")
        costs_matrix, coordenadas = self.make_costs_matrix()
        
        print("Calculando kernel...")
        K = self.get_kernel(costs_matrix)
        
        print("Ejecutando Sinkhorn...")
        P = self.sinkhorn(img1, img2, K, n_iter)
        
        print("Generando video...")
        frames = self.make_video(P, coordenadas, num_frames)
        
        return frames
    
    
    def run_color(self, n_iter= 100, num_frames= 60):

        R1 = self.img1[:, : , 2]
        R2 = self.img2[:, : , 2]
        G1 = self.img1[:, : , 1]
        G2 = self.img2[:, : , 1]
        B1 = self.img1[:, : , 0]
        B2 = self.img2[:, : , 0]

        imgs = [[B1, B2],[G1, G2],[R1, R2]]

        frames_colores = []

        for img1, img2 in imgs:

            frames_colores.append(self.run(img1, img2, n_iter, num_frames))

        frames_finales = []
        for f in range(num_frames):

            frames_finales.append(np.stack([frames_colores[0][f], frames_colores[1][f], frames_colores[2][f]], axis= 2))

        return frames_finales

    



if __name__ == '__main__':


    morphing = Morphing('caras/jesus.jpeg','caras/C3PO.jpeg', resolucion= 128, epsilon= 0.0005, n_iter= 300)
    frames = morphing.run_color(num_frames= 200, n_iter= 300)
    morphing.save_video(frames, output_path= 'morphing2.mp4')