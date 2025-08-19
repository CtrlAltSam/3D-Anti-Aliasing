import numpy as np
from transform import Transform
from stl import mesh as meshNp

class Mesh:
    def __init__(self, diffuse_color, specular_color, ka, kd, ks, ke):
       self.verts = []
       self.faces = []
       self.normals = []
       self.vertex_normals = []
       self.transform = Transform()
       self.diffuse_color = diffuse_color
       self.specular_color = specular_color
       self.ka = ka
       self.kd = kd
       self.ks = ks
       self.ke = ke

    def from_stl(stl_path, diffuse_color, specular_color, ka, kd, ks, ke):
        #Load STL file
        stl_mesh = meshNp.Mesh.from_file(stl_path)

        #Mesh Object to return
        mesh_obj = Mesh(diffuse_color, specular_color, ka, kd, ks, ke)

        #List to store unique vertices
        vertex_dict = {}
        vertex_list = []
        face_list = []
        
        #Go through each triangle in the STL mesh
        for i, triangle in enumerate(stl_mesh.vectors):
            face = []
            
            #For each vertex in the triangle, check if it's already in the vertex list
            for vertex in triangle:
                vertex_tuple = tuple(vertex)
                
                if vertex_tuple not in vertex_dict:
                    #If the vertex is not in the vertex_dict, add it
                    vertex_index = len(vertex_list)
                    vertex_dict[vertex_tuple] = vertex_index
                    vertex_list.append(vertex_tuple)
                else:
                    #Get the existing index of the vertex
                    vertex_index = vertex_dict[vertex_tuple]
                
                #Add vertex index to the face
                face.append(vertex_index)
            
            #Add the face to the face list
            face_list.append(face)
            
            #Add the normal for this face
            mesh_obj.normals.append(stl_mesh.normals[i])
        
        mesh_obj.verts = vertex_list
        mesh_obj.faces = face_list

        mesh_obj.compute_vertex_normals()

        return mesh_obj
    
    def compute_vertex_normals(self):
        """
        Compute the normals for each vertex by averaging the normals of adjacent faces.
        """
        # Initialize an array for vertex normals
        vertex_normals = np.zeros((len(self.verts), 3))

        # Loop through each face and accumulate the face normal for each vertex
        for face, normal in zip(self.faces, self.normals):
            for vertex_index in face:
                vertex_normals[vertex_index] += normal

        # Normalize the accumulated vertex normals
        vertex_normals = np.array([n / np.linalg.norm(n) if np.linalg.norm(n) > 0 else n 
                                   for n in vertex_normals])

        self.vertex_normals = vertex_normals.tolist()  # Store as a list for consistency