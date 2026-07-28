#!/bin/bash

pepsi_path=/users/t/j/tjaglal/Projects/iBME/Pepsi-SAXS
structures=$1
exp_path=$2
theta=$3  # Currently unused, unless iBME is activated
gl=$4 #Optional

gpdoc=$5
output_dir=$6
mkdir -p "$output_dir"
grep -v "#" "$exp_path" | awk '{print $1}' > "$output_dir/qvals.txt"


# Check if structure directory exists
if [[ ! -d "$structures" ]]; then
    echo "Error: structures not found: $structures"
    exit 1
fi

# Check if experimental data file exists
if [[ ! -f "$exp_path" ]]; then
    echo "Error: experimental data not found: $exp_path"
    exit 1
fi

# Get sorted list of structure files named mm016_*.pdb
structure_files=( $(find "$structures" -name "mm*.pdb" | sort) )
ens_size=${#structure_files[@]}

# Choose mode: single grid point or all
if [[ -n "$gl" ]]; then
    # ----------- SINGLE GRID POINT MODE -----------
    total_lines=$(grep -v "#" $5 | wc -l)
    if (( gl < 1 || gl > total_lines )); then
        echo "Error: Grid line $gl is out of range (1 to $total_lines)"
        exit 1
    fi
    grid_lines=($gl)
else
    # ----------- ALL GRID POINTS MODE -------------
    grid_lines=( $(seq 1 $(grep -v "#" $5 | wc -l)) )
fi

# Loop over selected grid point lines
for gl in "${grid_lines[@]}"; do
    grid_point=$(grep -v "#" $5 | sed -n "${gl}p" | awk '{print $1}')
    dro=$(grep -v "#" $5 | sed -n "${gl}p" | awk '{print $2}')
    r0=$(grep -v "#" $5 | sed -n "${gl}p" | awk '{print $3}')

    GP_DIR="$output_dir/GP$grid_point"
    echo "Running Pepsi-SAXS for grid point $grid_point (line $gl)..."
    mkdir -p "$GP_DIR"
  
    > "$GP_DIR/calc_saxs.txt"   # clear old data
    # Loop over structure files
    for i in "${!structure_files[@]}"; do
	echo "Processing structure $i: ${structure_files[$i]}"
        pdb_file="${structure_files[$i]}"
        if ! "$pepsi_path" "$pdb_file" "$exp_path" -o "$GP_DIR/saxs$i.dat" \
            -cst --cstFactor 0 --I0 1.0 --dro $dro \
            --r0_min_factor $r0 --r0_max_factor $r0 --r0_N 1; then
            echo "Pepsi failed for $pdb_file" >&2
            exit 1
        fi

        intensities=$(awk '!/^#/ {printf "%s ", $4}' "$GP_DIR/saxs$i.dat")
  echo "$i" "$intensities" >> "$GP_DIR/calc_saxs.txt"
        rm -f "$GP_DIR/saxs$i.dat"

        # Extract Rg only for grid_point == 0
        if [ "$grid_point" -eq 0 ]; then
            grep "Radius of gyration of the envelope" "$GP_DIR/saxs$i.log" | \
                awk '{print $8}' >> "$GP_DIR/Rg_env.dat" || true
        fi

        rm -f "$GP_DIR/saxs$i.log"
    done > "$GP_DIR/logPEPSI"

    # Move any generated iBME files (if run separately)
    #mv gp${grid_point}_* GP$grid_point/ 2>/dev/null
    #mv gp${grid_point}.log GP$grid_point/ 2>/dev/null
done
