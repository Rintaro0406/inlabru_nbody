#!/bin/bash

# Activate your Python environment if needed
# For example, uncomment and replace with your environment name:
# source ~/miniconda3/bin/activate your_environment_name

# Run the first Python script
echo "Running S_Spatial_Forest_v5.py..."
python /Users/r.kanaki/code/inlabru_nbody/code/S_Spatial_Forest_v5.py

# Check if the first script ran successfully
if [ $? -eq 0 ]; then
    echo "First script completed successfully."
else
    echo "First script encountered an error. Exiting..."
    exit 1
fi

# Run the second Python script
echo "Running RS_heatmap_plot_v4.py..."
python /Users/r.kanaki/code/inlabru_nbody/code/RS_heatmap_plot_v4.py

# Check if the second script ran successfully
if [ $? -eq 0 ]; then
    echo "Second script completed successfully."
else
    echo "Second script encountered an error. Exiting..."
    exit 1
fi

echo "All scripts completed successfully."