import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 

def plot_curve(points):
  x_coords, y_coords, z_coords = zip(*points)

  fig = plt.figure(figsize=(10, 8)) 
  ax = fig.add_subplot(111, projection='3d')

  ax.plot(x_coords, y_coords, z_coords,
          marker='o',          
          linestyle='-',       
          markersize=2,        
          label='Путь атомов')


  ax.scatter(x_coords[0], y_coords[0], z_coords[0], color='green', s=1, label='Начало')
  ax.scatter(x_coords[-1], y_coords[-1], z_coords[-1], color='red', s=1, label='Конец')

  ax.set_xlabel('Ось X')
  ax.set_ylabel('Ось Y')
  ax.set_zlabel('Ось Z')
  ax.set_title('Кривая на которой лежат CA  атомы')
  ax.legend() 
  ax.grid(True) 

  plt.show()

plot_curve(coords)