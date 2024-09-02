import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, LineString, Point, MultiPoint


def calculate_area(polygon):
    ''' Calcular el área de un polígono usando la fórmula de Shoelace '''
    x = polygon[:, 0]
    y = polygon[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

def calculate_longest_line(polygon):
    max_distance = 0
    longest_line = None
    for i in range(len(polygon)):
        for j in range(i + 1, len(polygon)):
            p1 = polygon[i]
            p2 = polygon[j]
            distance = np.linalg.norm(np.array(p1) - np.array(p2))
            if distance > max_distance:
                max_distance = distance
                longest_line = (p1, p2)
    return np.array(longest_line), np.array(max_distance)

def calculate_perpendicular_intersections(longest_line, polygon):
    # Calcular el punto medio de la línea más larga
    p1, p2 = longest_line
    midpoint = [(p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2]

    # Calcular la pendiente de la línea más larga
    if (p2[0] - p1[0]) != 0:
        slope = (p2[1] - p1[1]) / (p2[0] - p1[0])
    else:
        slope = float('inf')

    # La pendiente de la línea perpendicular
    if slope != 0 and slope != float('inf'):
        perpendicular_slope = -1 / slope
        perp_line = LineString([(midpoint[0] - 1000, midpoint[1] - 1000 * perpendicular_slope),
                                (midpoint[0] + 1000, midpoint[1] + 1000 * perpendicular_slope)])
    else:
        perp_line = LineString([(midpoint[0], midpoint[1] - 1000),
                                (midpoint[0], midpoint[1] + 1000)])

    # Encontrar la intersección de la línea perpendicular con el contorno del polígono
    poly = Polygon(polygon)
    intersection = perp_line.intersection(poly.boundary)

    # Manejar diferentes tipos de resultados de intersección
    intersection_points = []
    if isinstance(intersection, Point):
        intersection_points = [intersection]
    elif isinstance(intersection, MultiPoint):
        intersection_points = [point for point in intersection.geoms]
    elif isinstance(intersection, LineString):
        intersection_points = [Point(coords) for coords in intersection.coords]

    if len(intersection_points) == 0:
        raise ValueError("No se encontraron intersecciones entre la línea perpendicular y el contorno del lunar.")

    # Encontrar la distancia desde el punto medio a la intersección
    distances = [np.linalg.norm(np.array(midpoint) - np.array([point.x, point.y])) for point in intersection_points]
    max_distance = max(distances)

    return midpoint, intersection_points, max_distance

def plot_mole_axes(image_path, longest_line, midpoint, intersection, longest_line_cm, perp_line_cm):
    # Cargar la imagen
    img = plt.imread(image_path)

    # Configurar la figura
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(img)

    # Dibujar la línea longitudinal más larga
    p1, p2 = longest_line
    plt.plot([p1[0], p2[0]], [p1[1], p2[1]], color='blue', linewidth=2, label=f"Major axis: {longest_line_cm:.2f} cm")

    # Dibujar la línea perpendicular desde el punto medio
    for i, point in enumerate(intersection):
      if i == 0:  # Solo añadir la etiqueta en la primera iteración
          plt.plot([midpoint[0], point.x], [midpoint[1], point.y], color='red', linewidth=2, label=f"Minor axis: {perp_line_cm:.2f} cm")
      else:
          plt.plot([midpoint[0], point.x], [midpoint[1], point.y], color='red', linewidth=2)

    plt.legend()
    plt.axis('off')
    st.pyplot(fig)