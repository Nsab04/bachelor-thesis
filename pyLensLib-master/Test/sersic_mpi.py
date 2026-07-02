from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt

def sersic_profile(r, I_e, R_e, n):
    b_n = 2 * n - 1 / 3 + 0.009876 / n
    return I_e * np.exp(-b_n * ((r / R_e) ** (1 / n) - 1))

def compute_brightness_on_grid(start_row, end_row, grid_size, side_length, I_e, R_e, n):
    y, x = np.ogrid[start_row:end_row, :grid_size]
    center = grid_size // 2
    dx = side_length / grid_size  # Size of one grid cell in arcseconds

    # Convert grid indices to arcseconds from the center
    x = (x - center) * dx
    y = (y - center) * dx

    # Compute the radius from the center for each point
    r = np.sqrt(x**2 + y**2)

    # Compute the Sersic profile for each point
    return sersic_profile(r, I_e, R_e, n)

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    grid_size = 100000
    side_length = 10  # in arcseconds

    # Example parameters for the Sersic profile
    I_e = 1  # Example intensity at effective radius
    R_e = 1  # Example effective radius in arcseconds
    n = 4    # Sersic index

    # Determine the number of rows each process should compute
    rows_per_process = grid_size // size
    extra = grid_size % size

    start_row = rank * rows_per_process + min(rank, extra)
    end_row = start_row + rows_per_process + (1 if rank < extra else 0)

    # Compute the brightness for the assigned rows
    partial_result = compute_brightness_on_grid(start_row, end_row, grid_size, side_length, I_e, R_e, n)

    # Gather all partial results
    sendcounts = np.array(comm.gather(partial_result.size, 0))
    if rank == 0:
        full_result = np.empty([grid_size * grid_size], dtype=np.float64)
        displacements = np.hstack((0, np.cumsum(sendcounts)[:-1]))
    else:
        full_result = None
        displacements = None

    comm.Gatherv(sendbuf=partial_result.ravel(), recvbuf=[full_result, sendcounts, displacements, MPI.DOUBLE], root=0)

    # Root process can now handle the full result
    if rank == 0:
        full_result = full_result.reshape((grid_size, grid_size))
        fig,ax = plt.subplots(1,1,figsize=(10,10))
        ax.imshow(np.log10(full_result))
        plt.show()

if __name__ == '__main__':
    main()


