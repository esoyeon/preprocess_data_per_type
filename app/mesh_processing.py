import trimesh
import numpy as np
import io
import matplotlib.pyplot as plt

def preprocess_mesh(mesh_bytes: bytes) -> tuple:
    """
    Preprocess the 3D mesh:
    1. Load and normalize
    2. Remove duplicate vertices
    3. Fix mesh orientation
    4. Center the mesh
    """
    try:
        # Load mesh from bytes
        mesh = trimesh.load(io.BytesIO(mesh_bytes), file_type='off')
        
        # Normalize scale
        mesh.vertices -= mesh.center_mass
        mesh.vertices /= np.max(np.abs(mesh.vertices))
        
        # Remove duplicate vertices
        mesh.process()
        
        # Fix mesh orientation
        mesh.fix_normals()
        
        # Generate visualization
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_trisurf(mesh.vertices[:, 0], mesh.vertices[:, 1], mesh.vertices[:, 2],
                       triangles=mesh.faces, cmap='viridis')
        plt.title('Preprocessed Mesh')
        
        # Save visualization to bytes
        vis_buf = io.BytesIO()
        plt.savefig(vis_buf, format='png')
        plt.close()
        vis_buf.seek(0)
        
        # Save processed mesh to bytes
        mesh_buf = io.BytesIO()
        mesh.export(mesh_buf, file_type='off')
        
        return mesh_buf.getvalue(), vis_buf.getvalue()
        
    except Exception as e:
        print(f"Error processing mesh: {str(e)}")
        return mesh_bytes, None 