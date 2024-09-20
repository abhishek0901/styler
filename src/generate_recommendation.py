"""
Author : Abhishek Srivastava

Description : This file contains a standalone code to generate recommendations for outfits.
It will implement a greedy algorithm to find the best outfit to maximize utility(maximum summition of goodness score for n days).
These are the following constraints that this code needs to handle :-
1. Ensure plan is generated for n days
2. No 2 shits/pants are continued
3. If only_wear_once is True then if a shirt/pant is recommended once then don't recommend again
"""
from typing import Tuple
import numpy as np

def generate_recommendation(G:np.ndarray, n: int, only_wear_once: bool = False) -> np.ndarray[Tuple[int,int]]:
    pass


if __name__ == "__main__":
    print("Hello World!")