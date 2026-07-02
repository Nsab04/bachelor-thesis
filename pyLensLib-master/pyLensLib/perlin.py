import numpy as np
import noise

"""
This code utilizes the noise.pnoise2() function from the noise package to generate Perlin noise. 
The pnoise2() function takes in 2D coordinates (sample_x and sample_y) and generates Perlin noise 
at those coordinates. Adjust the parameters to fit your requirements. 

In the python noise module there are a few parameters that affect what you see when you generate your perlin noise:

scale: number that determines at what distance to view the noisemap.
octaves: the number of levels of detail you want you perlin noise to have.
lacunarity: number that determines how much detail is added or removed at each octave (adjusts frequency).
persistence: number that determines how much each octave contributes to the overall shape (adjusts amplitude).
We won’t worry about scale too much, you can use it to zoom out (bigger scale) or in (smaller scale).

Perlin noise combines multiple functions called ‘octaves’ to produce natural looking surfaces. 
Each octave adds a layer of detail to the surface. For example: octave 1 could be mountains, 
octave 2 could be boulders, octave 3 could be the rocks.

Lacunarity of more than 1 means that each octave will increase it’s level of fine grained detail 
(increased frqeuency). Lacunarity of 1 means that each octave will have the sam level of detail. 
Lacunarity of less than one means that each octave will get smoother. The last two are usually undesirable 
so a lacunarity of 2 works quite well.

Persistence determines how much each octave contributes to the overall structure of the noise map. 
If your persistence is 1 all octaves contribute equally. If you persistence is more than 1 sucessive octaves 
contribute more and you get something closer to regular noise (spoiler the regular noise image above is 
actually a perlin noise with a presistence of 5.0). A more default setting would be a presistance of less 
than 1.0 which will decrease the effect of later octaves.
"""


def generate_perlin_noise(width, height, scale=10, octaves=3, persistence=0.5, lacunarity=2.0, seed=None):
    """
    Generate a 2D Perlin noise map.

    Args:
        width (int): Width of the noise map.
        height (int): Height of the noise map.
        scale (float): Distance scale for viewing the noise map.
        octaves (int): Number of levels of detail in the noise.
        persistence (float): Amplitude contribution of each octave.
        lacunarity (float): Frequency multiplier for each octave.
        seed (int, optional): Random seed for reproducibility.

    Returns:
        ndarray: 2D array of normalized Perlin noise values in [0, 1].
    """
    if seed is not None:
        np.random.seed(seed)

    noise_map = np.zeros((height, width))
    for y in range(height):
        for x in range(width):
            value = 0
            for octave in range(octaves):
                octave_scale = scale * (lacunarity ** octave)
                sample_x = x / octave_scale
                sample_y = y / octave_scale
                value += noise.pnoise2(sample_x, sample_y, octaves=octaves) * persistence ** octave

            noise_map[y][x] = value

    # Normalize the values to range [0, 1]
    noise_map = (noise_map - np.min(noise_map)) / (np.max(noise_map) - np.min(noise_map))

    return noise_map
