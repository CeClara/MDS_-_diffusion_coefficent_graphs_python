import numpy as np
import matplotlib.pyplot as plt
import os
print( 'Current folder:',os.getcwd)
# Data
bsa = [0, 10, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200]
d = [0.4513, 0.3969, 0.3749, 0.3494, 0.3579, 0.2501,
     0.4458, 0.2798, 0.1884, 0.1884, 0.2351, 0.1198]

# Create figure
fig, ax = plt.subplots(figsize=(7.4, 4.8))

# Plot points
ax.scatter(bsa, d,
           marker='x',
           s=70,
           linewidths=1.9,
           color='black',
           label='Mean diffusion coefficient')

# Linear regression
slope, intercept = np.polyfit(bsa, d, 1)
x_fit = np.linspace(0, 200, 300)
ax.plot(x_fit, slope * x_fit + intercept,
        linestyle='--',
        linewidth=1.5,
        color='0.45',
        label='Linear regression')

# Title
fig.suptitle('Effect of BSA concentration on 1 μm bead diffusion coefficent',
             fontsize=13, y=0.985)

# Axis labels
ax.set_xlabel('BSA concentration (mg/ml)', fontsize=11, labelpad=10)
ax.set_ylabel('Diffusion coefficient, D (µm²/s)', fontsize=11, labelpad=14)

# Axes and ticks
ax.set_xlim(-5, 205)
ax.set_ylim(0, 0.60)
ax.set_xticks(np.arange(0, 201, 20))
ax.set_yticks(np.arange(0, 0.61, 0.1))

# Style
ax.tick_params(axis='both', labelsize=10, direction='out', width=1)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend(frameon=False, fontsize=9, loc='upper right')

# Layout spacing
fig.subplots_adjust(left=0.16, right=0.97, bottom=0.18, top=0.79)

# Save files
#plt.savefig('BSA_diffusion_publication_v2.png', dpi=1200, bbox_inches='tight')

plt.savefig( 'BSA_diffusion.pdf', bbox_inches= 'tight')

plt.show()
