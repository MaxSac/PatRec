import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D, proj3d  # IGNORE:unused-import
from matplotlib.patches import FancyArrowPatch
import numpy as np


class PCA(object):
    '''
    Einfaches 3D Beispiel fuer die Hauptkomponentenanalyse.
    '''

    def __init__(self, samples):
        #
        # Implementieren Sie den Konstruktor.
        # Bestimmen Sie hierfuer die Hauptachsen (principal components) der Daten
        # im Argument samples.
        #
        # Beachten Sie, dass spaeter die Methode transform_samples implementiert
        # werden soll, in der die PCA benutzt werden soll, um eine Dimensions-
        # reduktion durchfuehren zu koennen.
        # Fuehren Sie hier im Konstruktor diese Reduktion noch _NICHT_ durch,
        # speichern Sie aber alle notwendigen Variablen, die Sie fuer die Transformation
        # spaeter benoetigen, in entsprechenden Klassenvariablen.
        # Achten Sie darauf, dass alle Klassenvariablen sinnvoll belegt werden.

        self.samples = samples
        self.train_mean = np.mean(samples, axis=0)
        self.samples -= self.train_mean 
        self.train_cov = np.cov(self.samples, rowvar=0)
        self.eig_vals, self.eig_vecs = np.linalg.eig(self.train_cov)

        self.eig_vecs = self.eig_vecs[np.argsort(self.eig_vals)][::-1]
        self.eig_vals = self.eig_vals[np.argsort(self.eig_vals)][::-1]
        #raise NotImplementedError('Implement me')

    def transform_samples(self, samples, target_dim):
        if samples.shape[1] != self.eig_vecs.shape[0]:
            raise ValueError('Samples dimension does not match vector space transformation matrix')
        if target_dim < 1 or target_dim > samples.shape[1]:
            raise ValueError('Invalid target dimension')

        # Implementieren Sie die Dimensionsreduktion
        # Ueberlegen Sie, wie man die gesamte samples Matrix in einem transformiert (ohne Schleife)
        print self.eig_vecs.shape
        self.eig_vecs = self.eig_vecs[:,:target_dim]
        print self.eig_vecs.shape
        return np.dot(np.transpose(self.eig_vecs), np.transpose(samples))

        #raise NotImplementedError('Implement me')


    def plot_subspace(self, limits, color, linewidth, alpha, ellipsoid=True, coord_system=True, target_dim=None):
        center = self.train_mean
        radii = self.eig_vals
        rotation = self.eig_vecs.T


        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        if ellipsoid:
            self.plot_ellipsoid(center, radii, rotation,
                                color=color, linewidth=linewidth,
                                alpha=alpha, ax=ax)
        if coord_system:
            self.plot_coordinate_system(center, axes=rotation.T,
                                        axes_length=radii, ax=ax)
        self.set_axis_limits(ax, limits)

    @staticmethod
    def set_axis_limits(ax, limits):
        ax.set_xlim(limits[0][0], limits[0][1])
        ax.set_ylim(limits[1][0], limits[1][1])
        if len(limits) == 3:
            ax.set_zlim(limits[2][0], limits[2][1])

    @staticmethod
    def plot_sample_data(samples, color='b', annotations=None, ax=None):
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

        if samples.shape[1] == 2:
            xx = samples[:, 0]
            yy = samples[:, 1]
            ax.scatter(xx, yy, c=color, marker='o', alpha=1.0, edgecolor=(0, 0, 0))
        elif samples.shape[1] == 3:
            xx = samples[:, 0]
            yy = samples[:, 1]
            zz = samples[:, 2]
            ax.scatter(xx, yy, zz, c=color, marker='o', alpha=1.0, edgecolor=(0, 0, 0))

        if annotations is not None:
            for sample, annotation in zip(samples, annotations):
                sample_tup = tuple(sample)
                # * operator expands tuple in argument list
                ax.text(*sample_tup, s=annotation)  #pylint: disable=star-args

    @staticmethod
    def samples_coordinate_annotations(samples):
        samples_strcoords = [[str(coord) for coord in sample] for sample in samples]
        samples_labels = [' [ %s ]' % ', '.join(sample) for sample in samples_strcoords]
        return samples_labels

    @staticmethod
    def plot_ellipsoid(center, radii, rotation, color, linewidth, alpha, ax=None):
        if len(radii) != rotation.shape[0]:
            raise ValueError('Number of radii does not match rotation matrix')
        if len(radii) == 2:
            radii = list(radii) + [0.0]
            rotation = np.vstack((rotation, [0.0] * 3))
        u = np.linspace(0.0, 2.0 * np.pi, 100)
        v = np.linspace(0.0, np.pi, 100)
        x = radii[0] * np.outer(np.cos(u), np.sin(v))
        y = radii[1] * np.outer(np.sin(u), np.sin(v))
        z = radii[2] * np.outer(np.ones_like(u), np.cos(v))
        for i in range(len(x)):
            for j in range(len(x)):
                [x[i, j], y[i, j], z[i, j]] = np.dot([x[i, j], y[i, j], z[i, j]], rotation) + center

        # plot
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(x, y, z, rstride=4, cstride=4, color=color, alpha=alpha, linewidth=linewidth)

    @staticmethod
    def plot_coordinate_system(center, axes, axes_length, ax=None):
        # plot
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

        xs_center = [center[0]]
        ys_center = [center[1]]
        zs_center = [center[2]]
        for a, l in zip(axes.T, axes_length):
            p_axis = center + l * a
            xs = list(xs_center)
            ys = list(ys_center)
            zs = list(zs_center)
            xs.append(p_axis[0])
            ys.append(p_axis[1])
            zs.append(p_axis[2])
            arrow = Arrow3D(xs, ys, zs, lw=2)
            ax.add_artist(arrow)


class Arrow3D(FancyArrowPatch):

    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, _ = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)

