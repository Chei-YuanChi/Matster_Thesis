#!/bin/bash


# train for data after data preprocessing
# python Main.py --train True

# train for original data
# python Main.py --train True --origin_data True


# test 
# python Main.py --train False

# test (get omics weight)
# python Main.py --train False --get_weight omics

# test (get gene weight)
python Main.py --train False --get_weight gene --cell_line 25 --drug_index 26