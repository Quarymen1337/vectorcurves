def get_ca_coordinates(data):
  ca_coordinates = []
  for atom in data['chain']['atoms']:
    if atom['atom_name'] == 'CA':
      coordinates = (atom['x'], atom['y'], atom['z'])
      ca_coordinates.append(coordinates)
  return ca_coordinates