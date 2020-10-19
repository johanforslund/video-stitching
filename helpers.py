import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch 

def show_corresp(img1, img2, p1, p2, vertical=True):
    """Draw point correspondences

    Draws lines between corresponding points in two images.
    Hovering over a line highlights that line.

    Note that you need to call plt.show()
    after calling this function to view the results.


    Parameters
    ---------------
    img1 : (M, N) array
        First image
    img2 : (M, N) array
        Second image
    p1 : (2, K) array
        Points in first image
    p2 : (2, K) array
        Points in second image
    vertical: bool
        Controls layout of the images

    Returns
    ------------
    fig : Figure
        The drawn figure
    """
    assert p1.shape == p2.shape

    fig = plt.figure()
    plt.gray()
    rows, cols = (2, 1) if vertical else (1, 2)
    ax_left = fig.add_subplot(rows, cols, 1)
    ax_right = fig.add_subplot(rows, cols, 2)

    imshow_args = {'interpolation': 'nearest'}
    im_left_artist = ax_left.imshow(img1, **imshow_args)
    im_right_artist = ax_right.imshow(img2, **imshow_args)

    connection_patches = []

    corr_data = {
        'active': None,
        'patches': connection_patches
    }

    def hover_cp(event):
        if corr_data['active'] is not None:
            if corr_data['active'].contains(event, radius=10)[0] == True:
                return
            else:
                plt.setp(corr_data['active'], color='b')
                plt.draw()
                corr_data['active'] = None

        for cp in corr_data['patches']:
            contained, cdict = cp.contains(event, radius=10)
            if contained == True:
                corr_data['active'] = cp
                plt.setp(cp, color='r')
                break
        plt.draw()

    for (xyA, xyB) in zip(p1.T, p2.T):
        cp = ConnectionPatch(xyA=xyB, xyB=xyA,
                             coordsA='data', coordsB='data',
                             axesA=ax_right, axesB=ax_left,
                             arrowstyle='-', color='b')
        connection_patches.append(cp)
        ax_right.add_artist(cp)

    ax_right.set_zorder(ax_left.get_zorder() + 1)
    ax_left.plot(p1[0], p1[1], 'o')
    ax_right.plot(p2[0], p2[1], 'o')

    for im, ax in ((img1, ax_left), (img2, ax_right)):
        ax.set_xlim(0, im.shape[1]-1)
        ax.set_ylim(im.shape[0]-1, 0)

    fig.canvas.mpl_connect('motion_notify_event', hover_cp)
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05,
                        wspace=0.05, hspace=0.05)

    return fig