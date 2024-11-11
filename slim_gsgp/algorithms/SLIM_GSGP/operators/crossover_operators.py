# MIT License
#
# Copyright (c) 2024 sofiaccap
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
Geometric crossover implementation for genetic programming trees.
"""
import random

import torch
from slim_gsgp.algorithms.GSGP.representations.tree import Tree
from slim_gsgp.algorithms.SLIM_GSGP.representations.individual import Individual
from slim_gsgp.utils.utils import get_random_tree

def geometric_crossover(FUNCTIONS, TERMINALS,CONSTANTS):
    """
    Generate an inflate mutation function.

    Parameters
    ----------
    FUNCTIONS : dict
        The dictionary of functions used in the mutation.
    TERMINALS : dict
        The dictionary of terminals used in the mutation.
    CONSTANTS : dict
        The dictionary of constants used in the mutation.
    Returns
    -------
    Callable
        An geometric crossover function (`geom_xo`).

        Parameters
        ----------
        individual1: Individual
            The tree 1 Individual to crossover.
        individual2: Individual
            The tree 2 Individual to crossover.
        X : torch.Tensor
            Input data for calculating semantics.
        max_depth : int, optional
            Maximum depth for generated trees (default: 8).
        p_c : float, optional
            Probability of choosing constants (default: 0.1).
        X_test : torch.Tensor, optional
            Test data for calculating test semantics (default: None).
        grow_probability : float, optional
            Probability of growing trees during mutation (default: 1).
        reconstruct : bool, optional
            Whether to reconstruct the Individual's collection after mutation (default: True).

        Returns
        -------
        Individual
            The geometric crossover tree Individual.
    """
    def geom_xo(
        individual1,
        individual2,
        X,
        max_depth=8,
        p_c=0.1,
        X_test=None,
        grow_probability=1,
        reconstruct=True,
    ):
        """
        Perform geometric crossover on the given Individuals.

        Parameters
        ----------
        individual1 : Individual
            The tree 1 Individual to crossover.
        individual2 : Individual
            The tree 2 Individual to crossover.
        X : torch.Tensor
            Input data for calculating semantics.
        max_depth : int, optional
            Maximum depth for generated trees (default: 8).
        p_c : float, optional
            Probability of choosing constants (default: 0.1).
        X_test : torch.Tensor, optional
            Test data for calculating test semantics (default: None).
        grow_probability : float, optional
            Probability of growing trees during mutation (default: 1).
        reconstruct : bool, optional
            Whether to reconstruct the Individual's collection after crossover (default: True).

        Returns
        -------
        Individual
            The geometric crossover tree Individual.
        """
        # getting one random tree
        random_tree1 = get_random_tree(
                max_depth,
                FUNCTIONS,
                TERMINALS,
                CONSTANTS,
                inputs=X,
                p_c=p_c,
                grow_probability=grow_probability,
                logistic = False
                #logistic = True #in doubt here
            )

        # calculating the semantics of the random tree on testing, if applicable
        if X_test is not None:
                    random_tree1.calculate_semantics(
                        X_test, testing=True, #logistic=single_tree_sigmoid or sig
                    )

        #since we want to keep working with trees, we need to create tree instances for each new tree that arises from geometric crossover
        #left block
        left_tree = Tree(
              structure = [torch.mul, *individual1.collection, random_tree1],
              train_semantics= torch.mul(*individual1.train_semantics, random_tree1.train_semantics),
              test_semantics= (torch.mul(*individual1.test_semantics, random_tree1.test_semantics) 
               if individual1.test_semantics is not None 
                else None),
             reconstruct=True,
        ) 
        #nested tree of rigth block
        nested_tree = Tree(
              structure = [torch.sub, 1, random_tree1],
              train_semantics= torch.sub(1, random_tree1.train_semantics),
              test_semantics= (torch.sub(1, random_tree1.test_semantics)
               if random_tree1.test_semantics is not None 
                else None),
             reconstruct=True,
        ) 

        #rigth block
        rigth_tree = Tree(
              structure = [torch.mul, nested_tree, *individual2.collection ],
              train_semantics= torch.mul(nested_tree.train_semantics, *individual2.train_semantics),
              test_semantics= (torch.mul(nested_tree.test_semantics, *individual2.test_semantics)
               if individual2.test_semantics is not None 
                else None),
             reconstruct=True,
        ) 

        #final tree
        new_tree =  Tree(
              structure = [torch.add, left_tree, rigth_tree],
              train_semantics= torch.add(left_tree.train_semantics, rigth_tree.train_semantics),
              test_semantics= (torch.add(left_tree.test_semantics, rigth_tree.test_semantics)
               if left_tree.test_semantics is not None 
                else None),
             reconstruct=True,
        ) 

        offs = Individual(
            collection=( [new_tree] 
            if reconstruct
            else None
        ),
        train_semantics= new_tree.train_semantics,
        test_semantics=(new_tree.test_semantics
                
                if new_tree.test_semantics is not None  
                else None
         ),
            reconstruct=reconstruct,
        )

        offs.size = len(offs.collection)
        offs.nodes_collection = [tree.nodes for tree in offs.collection]
        offs.nodes_count = sum(offs.nodes_collection) + (offs.size - 1)

        offs.depth_collection = [tree.depth for tree in offs.collection]
        offs.depth = max(
          [
                depth - (i - 1) if i != 0 else depth
                for i, depth in enumerate(offs.depth_collection)
            ]
        ) + (offs.size - 1)
        #------
        #i am now thinking about the representation of geometric_xo better and am extremelly lost
        #i need to know how geometric_xo works on SLIM: 1) do wee add a new tree to 1 parent; 2) do we select a tree from the parent; 
        #3) do we perform xo to every tree in the individual and keep the smallest individual
        return offs

    return geom_xo
      
def swap_base_crossover(individual1, individual2, reconstruct = True ):
    """
    Performs crossover between two individuals by swapping the first block of them.

    Parameters
    ----------
    individual1 : Tree
        The first parent individual.
    individual2 : Tree
        The second parent individual.
    reconstruct : bool, optional
        Whether to reconstruct the Individual's collection after crossover (default: True).

    Returns
    -------
    Individual
            The base swaped crossover tree Individuals.
    """

    # changing the first block of two individuals will lead to two new individuals
    offs1 = Individual(collection=(
            [
                *individual2.collection[:1],
                *individual1.collection[1:],
            ]
            if reconstruct
            else None
        ),
         train_semantics=torch.stack(
            [
                *individual2.train_semantics[:1],
                *individual1.train_semantics[1:],
            ]
        ),
        test_semantics=(
            torch.stack(
                [
                    *individual2.test_semantics[:1],
                    *individual1.test_semantics[1:],
                ]
            )
            if individual1.test_semantics is not None and individual2.test_semantics is not None
            else None
        ),
        reconstruct=reconstruct,
        
        )
    
    offs2 = Individual(collection = ( 
            [
                *individual1.collection[:1],
                *individual2.collection[1:],
            ] 
            if reconstruct
            else None
            ),
             train_semantics=torch.stack(
            [
                *individual1.train_semantics[:1],
                *individual2.train_semantics[1 :],
            ]
        ),
        test_semantics=(
            torch.stack(
                [
                    *individual1.test_semantics[:1],
                    *individual2.test_semantics[1 :],
                ]
            )
            if individual1.test_semantics is not None and individual2.test_semantics is not None
            else None
        ),
        reconstruct=reconstruct,
            
            )
    
    # computing offspring attributes
    offs1.size = individual1.size 
    offs2.size = individual2.size 

    offs1.nodes_collection = [
        *individual2.nodes_collection[:1],
        *individual1.nodes_collection[1:],
    ]

    offs2.nodes_collection = [
        *individual1.nodes_collection[:1],
        *individual2.nodes_collection[1:],
    ]

   
    offs1.nodes_count = sum(offs1.nodes_collection) + (offs1.size - 1)
    offs2.nodes_count = sum(offs2.nodes_collection) + (offs2.size - 1)

    offs1.depth_collection = [
        *individual2.depth_collection[:1],
        *individual1.depth_collection[1:],
    ]

    offs2.depth_collection = [
        *individual1.depth_collection[:1],
        *individual2.depth_collection[1:],
    ]

    offs1.depth = max(
        [
            depth - (i - 1) if i != 0 else depth
            for i, depth in enumerate(offs1.depth_collection)
        ]
    ) + (offs1.size - 1)

    offs2.depth = max(
        [
            depth - (i - 1) if i != 0 else depth
            for i, depth in enumerate(offs2.depth_collection)
        ]
    ) + (offs2.size - 1)

    return offs1, offs2