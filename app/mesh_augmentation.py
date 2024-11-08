import trimesh
import numpy as np
import io
import matplotlib.pyplot as plt
import random

def random_rotation(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Apply random rotation"""
    angle = random.uniform(0, 2 * np.pi)
    axis = random.choice(['x', 'y', 'z'])
    matrix = trimesh.transformations.rotation_matrix(angle, [1, 0, 0] if axis == 'x' else [0, 1, 0] if axis == 'y' else [0, 0, 1])
    mesh.apply_transform(matrix)
    return mesh

def random_scale(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Apply random scaling"""
    scale = random.uniform(0.8, 1.2)
    matrix = trimesh.transformations.scale_matrix(scale)
    mesh.apply_transform(matrix)
    return mesh

def add_noise(mesh: trimesh.Trimesh, noise_factor: float = 0.02) -> trimesh.Trimesh:
    """Add random noise to vertices"""
    noise = np.random.normal(0, noise_factor, mesh.vertices.shape)
    mesh.vertices += noise
    return mesh

def augment_mesh(mesh_bytes: bytes) -> tuple:
    """
    Augment the 3D mesh:
    1. Random rotation
    2. Random scaling
    3. Add vertex noise
    """
    try:
        # Load mesh from bytes
        mesh = trimesh.load(io.BytesIO(mesh_bytes), file_type='off')
        
        # Apply augmentations
        mesh = random_rotation(mesh)
        mesh = random_scale(mesh)
        mesh = add_noise(mesh)
        
        # Generate visualization
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_trisurf(mesh.vertices[:, 0], mesh.vertices[:, 1], mesh.vertices[:, 2],
                       triangles=mesh.faces, cmap='viridis')
        plt.title('Augmented Mesh')
        
        # Save visualization to bytes
        vis_buf = io.BytesIO()
        plt.savefig(vis_buf, format='png')
        plt.close()
        vis_buf.seek(0)
        
        # Save augmented mesh to bytes
        mesh_buf = io.BytesIO()
        mesh.export(mesh_buf, file_type='off')
        
        return mesh_buf.getvalue(), vis_buf.getvalue()
        
    except Exception as e:
        print(f"Error augmenting mesh: {str(e)}")
        return mesh_bytes, None 