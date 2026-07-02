from skimage.transform import  resize
import numpy as np

class raymesh(object):
    """
    Class for generating and refining a mesh of rays for lensing simulations.

    Attributes:
        nray_ini (int): Initial number of rays per axis.
        rule (ndarray): Rule map for refinement.
        nref (int): Number of refinement steps.
        px, py (ndarray): Final ray positions.
        refinement (ndarray): Refinement labels for each ray.
    """
    def __init__(self,nray_ini=2048,rule=np.ones((2048,2048)),nref=2):
        """
        Initialize a raymesh object and generate a refined mesh of rays.

        Args:
            nray_ini (int): Initial number of rays per axis.
            rule (ndarray): Rule map for refinement.
            nref (int): Number of refinement steps.

        Returns:
            None
        """
        self.nray_ini = nray_ini
        self.rule = rule
        self.nref=nref

        if self.rule.shape[0] != nray_ini:
            self.rule = resize(rule, (nray_ini, nray_ini), anti_aliasing=True)
            print ("Inconsitent size of rule map; rule map has been rescaled")

        # starting grid
        pixel_s = 1
        xx = np.arange(0, self.rule.shape[0], pixel_s)
        px, py = np.meshgrid(xx, xx)
        px_ = px.copy().flatten()
        py_ = py.copy().flatten()
        ref_label = np.zeros_like(px_)

        px_fin = px_
        py_fin = py_
        labels = ref_label

        cur_shape = self.rule.shape[0]
        pixel_s = 2.0
        for i in range(nref):
            # step 1: compute gradient of the rule
            rule_ = resize(rule, (cur_shape, cur_shape), anti_aliasing=True)
            pixel_s = pixel_s / 2.0
            xx = np.arange(0, self.rule.shape[0], pixel_s)
            px, py = np.meshgrid(xx, xx)
            px_ = px.flatten()
            py_ = py.flatten()
            grad_y, grad_x = np.gradient(rule_)
            grad_ = np.sqrt(grad_x ** 2 + grad_y ** 2)

            # step 2: select pixels where gradient is >1
            isel = (np.abs(grad_) > 1.0)
            isel_ = isel.flatten()
            px_fin = np.append(px_fin, px_[isel_])
            py_fin = np.append(py_fin, py_[isel_])
            ref_label = np.zeros_like(px_) + i
            labels = np.append(labels, ref_label[isel_])

            # update size for the next iteration
            cur_shape = cur_shape * 2

        # step 3: delete repeated rays. If a ray is repeated, keep the latest addition
        px_fin = np.flip(px_fin)
        py_fin = np.flip(py_fin)
        labels = np.flip(labels)
        stack = np.stack((px_fin, py_fin), axis=-1)
        u, i_unique = np.unique(stack, axis=0, return_index=True)
        self.px=px_fin[i_unique]
        self.py=py_fin[i_unique]
        self.refinement=labels[i_unique]
