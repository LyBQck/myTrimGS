import numpy as np

def get_rotation(R, theta, phi, alpha):
    rot_phi = lambda phi : np.array([
            [1,0,0],
            [0,np.cos(phi),-np.sin(phi)],
            [0,np.sin(phi), np.cos(phi)]])
    
    rot_theta = lambda th : np.array([
        [np.cos(th),0,-np.sin(th)],
        [0,1,0],
        [np.sin(th),0, np.cos(th)]])
    
    rot_alpha = lambda al : np.array([
        [np.cos(al), -np.sin(al), 0],
        [np.sin(al), np.cos(al), 0],
        [0, 0, 1]
    ])
    def pose_spherical(Rot):
        Rot = Rot.copy() # R is also okay
        Rot = rot_theta(-theta/180.*np.pi) @ Rot
        Rot = rot_phi(-phi/180.*np.pi) @ Rot
        Rot = rot_alpha(-alpha/180. * np.pi) @ Rot
        return Rot
    
    return pose_spherical(R)

def get_transform(w2c, trans, rots):
    c2w = np.linalg.inv(w2c)
    R = get_rotation(c2w[:3, :3], rots[0], rots[1], rots[2])
    T = c2w[:3, 3] + trans
    c2w[:3, :3] = R
    c2w[:3, 3] = T
    return np.linalg.inv(c2w)