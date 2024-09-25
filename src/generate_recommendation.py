"""
================================================

Author : Abhishek Srivastava

Description : 
This file contains a standalone code to generate recommendations for outfits.
It will implement a greedy algorithm to find the best outfit to maximize utility(maximum summition of goodness score for n days).
These are the following constraints that this code needs to handle :-
1. Ensure plan is generated for n days
2. No 2 shirts/pants are continued
3. If only_wear_once is True then if a shirt/pant is recommended once then don't recommend again

================================================
"""
from typing import Tuple
import numpy as np
from dataclasses import dataclass
from collections import defaultdict
import copy

# TODO: Find better ways to calculate this value.
MIN_GOODNESS = 0.01

@dataclass
class Outfit:
    shirt_index: int
    pant_index: int
    score: float

def _calculate_score(goodness: float, shirt_utility: int, pant_utility: int):
    return goodness * (shirt_utility + pant_utility)

def generate_recommendation(G:np.ndarray, U:np.ndarray, n: int, only_wear_once: bool = False) -> np.ndarray[Tuple[int,int]]:
    """
        Generate recommendation for user.

        Parameters
        ----------
        G : np.ndarray
            A goodness matrix that has score for each of the outfit combination. It's a P x P matrix that stores goodness of each combination. 
            Assume G[i][i] = -inf, can't recommend same shirt/pant to wear unless it's a dress.
            Another extension of this problem is to include multiple wears.
        U: np.ndarray
            Utility of each of the cloth. It can be (1/2)**num_times_wore. 
        n : int
            The number of days for which schedule should be generated.
        only_wear_once: bool
            If it's true then don't repeat recommended shirt/pant again.

        Returns
        -------
        np.ndarray
            An array of tuples for respective (i,j) for chosen outfit.
    """

    # Generate Tuple(i,j) -> goodness score list
    num_clothes = G.shape[0]
    if num_clothes == 0:
        raise ValueError("num_clothes should be non zero.")
    
    goodness_list: list[Outfit] = [Outfit(shirt_index=i,pant_index=j,score=_calculate_score(goodness_list[i,j], U[i], U[j])) for i in num_clothes for j in num_clothes]

    # Sort list
    goodness_list = sorted(goodness_list, key=lambda x: x.score, reverse=True)

    # Filter any values which are below min thrshold
    first_key = -1
    for i,outfit in enumerate(goodness_list):
        if outfit.score < MIN_GOODNESS:
            first_key = i
            break
    
    goodness_list = goodness_list[:first_key]
    
    explored_cloth = defaultdict(lambda : False)
    valid_index = {i:True for i in range(len(goodness_list))}
    result = []

    current_index = 0

    # First try to utilize maximum cloth and then repeat if it permits
    while len(result) != n:
        current_outfit = goodness_list[current_index]
        result.append((current_outfit.shirt_index, current_outfit.pant_index))
        explored_cloth[current_outfit.shirt_index] = True
        explored_cloth[current_outfit.pant_index] = True
        valid_index[current_index] = False
        current_index_updated = False

        # Find next index which is unexplored
        for i in range(current_index + 1, len(goodness_list)):
            current_outfit = goodness_list[i]
            if explored_cloth[current_outfit.shirt_index] == False and explored_cloth[current_outfit.pant_index] == False:
                current_index = i
                current_index_updated = True
                break

        # no cloth left to explore
        if not current_index_updated:
            for j in range(len(valid_index)):
                if valid_index[j]:
                    current_index = j
                    current_index_updated = True
        
        # If nothing to explore then repeat
        if not current_index_updated:
            curr_length = len(result)
            temp_recommendation = copy(result)
            current_index = 0
            while len(result) != n:
                result.append(temp_recommendation[current_index])
                current_index = (current_index + 1) % curr_length


    return result


if __name__ == "__main__":
    print("Hello World!")