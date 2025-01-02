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
import numpy as np
from slim_gsgp.algorithms.GSGP.representations.tree import Tree
from slim_gsgp.algorithms.SLIM_GSGP.representations.individual import Individual
from slim_gsgp.utils.utils import get_random_tree

def geometric_crossover(FUNCTIONS, TERMINALS,CONSTANTS):
    """
    Generate an geometric crossover function.

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

def donor_xo(individual1, individual2, reconstruct = True):
    """
    Performs crossover between 2 individuals by donating a block of the donor to the reciever (selected at random between 2 individuals)

    Parameters
    ----------
    individual1 : Tree
        The first individual.
    individual2: list 
        The second individual.
    reconstruct : bool, optional
        Whether to reconstruct the Individual's collection after crossover (default: True).

    Returns
    -------
    Individuals
            The donor_gxo crossover tree Individuals.
    """
      
    #choose at random the index of the donor (0 = individual1, 1= individual2)
    donor_idx = random.randint(0,1)
    donor = individual1 if donor_idx == 0 else individual2
    recipient = individual1 if donor_idx == 1 else individual2

    #choose at random a block from the donor
    if donor.size > 1:
        donation_idx = random.randint(1, donor.size - 1 )

        recipient_offs = Individual(collection=[*recipient.collection, donor[donation_idx]]
                         if reconstruct else None,
                          train_semantics=torch.stack([*recipient.train_semantics, donor.train_semantics[donation_idx]]),
                          test_semantics=torch.stack([*recipient.test_semantics, donor.test_semantics[donation_idx]])
                          if donor.test_semantics is not None
                          else None,
                          reconstruct=reconstruct)

        donor_offs = Individual(collection=donor.collection[:donation_idx] + donor.collection[donation_idx+1:]
                         if reconstruct else None,
                          train_semantics=torch.stack(
                                [*donor.train_semantics[:donation_idx], *donor.train_semantics[donation_idx + 1:]]),
                          test_semantics=torch.stack(
                [*donor.test_semantics[:donation_idx], *donor.test_semantics[donation_idx + 1:]])
                          if donor.test_semantics is not None
                          else None,
                          reconstruct=reconstruct)

        recipient_offs.nodes_collection = recipient.nodes_collection + [donor.nodes_collection[donation_idx]]
        donor_offs.nodes_collection = donor.nodes_collection[:donation_idx] + donor.nodes_collection[donation_idx + 1:]

        recipient_offs.depth_collection = recipient.depth_collection + [donor.depth_collection[donation_idx]]
        donor_offs.depth_collection = donor.depth_collection[:donation_idx] + donor.depth_collection[donation_idx + 1:]

        recipient_offs.size = recipient.size + 1
        donor_offs.size = donor.size - 1

        recipient_offs.nodes_count = sum(recipient_offs.nodes_collection) + (recipient_offs.size - 1)
        donor_offs.nodes_count = sum(donor_offs.nodes_collection) + (donor_offs.size - 1)

        recipient_offs.depth = max([ depth - (i - 1) if i != 0 else depth
                                    for i, depth in enumerate(recipient_offs.depth_collection)]
                                    ) + (recipient_offs.size - 1)
        
        donor_offs.depth = max([depth - (i - 1) if i != 0 else depth
                                for i, depth in enumerate(donor_offs.depth_collection)]
                            ) + (donor_offs.size - 1)

        if donor_idx == 0:
            return donor_offs, recipient_offs
        else:
            return recipient_offs, donor_offs

    else:

        return individual1, individual2
    
def donor_n_xo(n= 5):
    """
    Generate a donor_n_gxo function.

    Parameters
    ----------
    n: int
        Number of individuals to be used in crossover (1 donor, n-1 recievers).
    Returns
    -------
    Callable
        An donor_n_gxo crossover function (`d_n_xo`).

        Performs crossover between n individuals by donating a block of the donor to the n individuals in the reciever_list

        Parameters
        ----------
        individual_list: list 
            A list of Trees with n individuals to be used on donor_n_gxo.
        reconstruct : bool, optional
            Whether to reconstruct the Individual's collection after crossover (default: True).

        Returns
        -------
        Individuals
            The donor_n_gxo crossover tree n Individuals.
    """
    def d_n_xo(individual_list, reconstruct = True):
    #In this crossover, we recieve one individual that is the parent and a list of individuals that are recievers
    #the goal is to perform the same operation but instead of having one reciever, having n recievers

        #choose at random the index of the donor (0 = p1, 1= p2)
        donor_idx = random.randint(0,n-1)
        donor = individual_list[donor_idx]
        recipients = individual_list[:donor_idx]
        recipients.extend(individual_list[donor_idx+1:])

        #choose at random a block from the donor
        if donor.size > 1:
            donation_idx = random.randint(1, donor.size - 1 )

            xo_recipients = ()

            for recipient in recipients:
                recipient_offs = Individual(collection=[*recipient.collection, donor[donation_idx]]
                         if reconstruct else None,
                          train_semantics=torch.stack([*recipient.train_semantics, donor.train_semantics[donation_idx]]),
                          test_semantics=torch.stack([*recipient.test_semantics, donor.test_semantics[donation_idx]])
                          if donor.test_semantics is not None
                          else None,
                          reconstruct=reconstruct)
                recipient_offs.nodes_collection = recipient.nodes_collection + [donor.nodes_collection[donation_idx]]
                recipient_offs.depth_collection = recipient.depth_collection + [donor.depth_collection[donation_idx]]
                recipient_offs.size = recipient.size + 1
                recipient_offs.nodes_count = sum(recipient_offs.nodes_collection) + (recipient_offs.size - 1)
                recipient_offs.depth =  max([ depth - (i - 1) if i != 0 else depth
                                            for i, depth in enumerate(recipient_offs.depth_collection)]
                                            ) + (recipient_offs.size - 1)

                xo_recipients += xo_recipients + (recipient_offs,)
                #xo_recipients.extend(recipient_offs)

            donor_offs = Individual(collection=donor.collection[:donation_idx] + donor.collection[donation_idx+1:]
                         if reconstruct else None,
                          train_semantics=torch.stack(
                                [*donor.train_semantics[:donation_idx], *donor.train_semantics[donation_idx + 1:]]),
                          test_semantics=torch.stack(
                [*donor.test_semantics[:donation_idx], *donor.test_semantics[donation_idx + 1:]])
                          if donor.test_semantics is not None
                          else None,
                          reconstruct=reconstruct)

            donor_offs.nodes_collection = donor.nodes_collection[:donation_idx] + donor.nodes_collection[donation_idx + 1:]
            donor_offs.depth_collection = donor.depth_collection[:donation_idx] + donor.depth_collection[donation_idx + 1:]
            donor_offs.size = donor.size - 1
            donor_offs.nodes_count = sum(donor_offs.nodes_collection) + (donor_offs.size - 1)
            donor_offs.depth = max([depth - (i - 1) if i != 0 else depth
                                for i, depth in enumerate(donor_offs.depth_collection)]
                            ) + (donor_offs.size - 1)

            #xo_recipients += (donor_offs,)
            xo_recipients = xo_recipients[:donation_idx] + (donor_offs,) + xo_recipients[donor_idx:]

            return *xo_recipients,

        else:
            return *individual_list,
    return d_n_xo

def best_donor_xo(measure = 'min_fitness'):
    """
    Generate a best_donor_xo function.

    Parameters
    ----------
    measure: string
        Measure to be considered when selection the 'best' individual to be donor.
    Returns
    -------
    Callable
        An best_donor_xo crossover function (`best_d_xo`).

        Performs crossover between 2 individuals by donating a block of the donor to the reciever.
        The donor is selected based on which individual is the 'best' in terms of the selected measure.

        Parameters
        ----------
        individual1 : Tree
            The first individual.
        individual2: list 
            The second individual.
        reconstruct : bool, optional
            Whether to reconstruct the Individual's collection after crossover (default: True).

        Returns
        -------
        Individuals
            The best_donor_xo crossover tree Individuals.
    """
    def best_d_xo(individual1, individual2, reconstruct = True):
         
    #choose the best individual
        if measure == 'biggest':
            donor_idx = np.argmax([ind.size for ind in [individual1, individual2]])
        elif measure == 'min_fitness':
            donor_idx = np.argmin([ind.fitness for ind in [individual1, individual2]])
        elif measure == 'max_fitness':
            donor_idx = np.argmax([ind.fitness for ind in [individual1, individual2]])
    
        donor = individual1 if donor_idx == 0 else individual2
        recipient = individual1 if donor_idx == 1 else individual2

    #choose at random a block from the donor
        if donor.size > 1:
            donation_idx = random.randint(1, donor.size - 1 )

            recipient_offs = Individual(collection=[*recipient.collection, donor[donation_idx]]
                         if reconstruct else None,
                          train_semantics=torch.stack([*recipient.train_semantics, donor.train_semantics[donation_idx]]),
                          test_semantics=torch.stack([*recipient.test_semantics, donor.test_semantics[donation_idx]])
                          if donor.test_semantics is not None
                          else None,
                          reconstruct=reconstruct)

            donor_offs = Individual(collection=donor.collection[:donation_idx] + donor.collection[donation_idx+1:]
                         if reconstruct else None,
                          train_semantics=torch.stack(
                                [*donor.train_semantics[:donation_idx], *donor.train_semantics[donation_idx + 1:]]),
                          test_semantics=torch.stack(
                [*donor.test_semantics[:donation_idx], *donor.test_semantics[donation_idx + 1:]])
                          if donor.test_semantics is not None
                          else None,
                          reconstruct=reconstruct)

            recipient_offs.nodes_collection = recipient.nodes_collection + [donor.nodes_collection[donation_idx]]
            donor_offs.nodes_collection = donor.nodes_collection[:donation_idx] + donor.nodes_collection[donation_idx + 1:]

            recipient_offs.depth_collection = recipient.depth_collection + [donor.depth_collection[donation_idx]]
            donor_offs.depth_collection = donor.depth_collection[:donation_idx] + donor.depth_collection[donation_idx + 1:]

            recipient_offs.size = recipient.size + 1
            donor_offs.size = donor.size - 1

            recipient_offs.nodes_count = sum(recipient_offs.nodes_collection) + (recipient_offs.size - 1)
            donor_offs.nodes_count = sum(donor_offs.nodes_collection) + (donor_offs.size - 1)

            recipient_offs.depth = recipient_offs.depth = max([ depth - (i - 1) if i != 0 else depth
                                                                    for i, depth in enumerate(recipient_offs.depth_collection)]
                                                                    ) + (recipient_offs.size - 1)
            donor_offs.depth = max([depth - (i - 1) if i != 0 else depth
                                for i, depth in enumerate(donor_offs.depth_collection)]
                            ) + (donor_offs.size - 1)

            if donor_idx == 0:
                return donor_offs, recipient_offs
            else:
                return recipient_offs, donor_offs

        else:

            return individual1, individual2
    
    return best_d_xo 

def best_donor_n_xo(n= 5, measure = 'min_fitness'):
    """
    Generate a best_donor_n_xo function.

    Parameters
    ----------
    n: int
        Number of individuals to be used in crossover (1 donor, n-1 recievers).
    measure: string
        Measure to be considered when selection the 'best' individual to be donor.

    Returns
    -------
    Callable
        An best_donor_n_xo crossover function (`best_d_n_xo`).

        Performs crossover between n individuals by donating a block of the donor to the n individuals in the reciever_list

        Parameters
        ----------
        individual_list: list 
            A list of Trees with n individuals to be used on donor_n_gxo.
        reconstruct : bool, optional
            Whether to reconstruct the Individual's collection after crossover (default: True).

        Returns
        -------
        Individuals
            The best_donor_n_xo crossover tree n Individuals.
    """
    def best_d_n_xo(individual_list, reconstruct = True):

        _ = n #dummy variable so n is identifiable

        #choose best individual to be donor:
        if measure == 'biggest':
            donor_idx = np.argmax([ind.size for ind in individual_list])
        elif measure == 'min_fitness':
            donor_idx = np.argmin([ind.fitness for ind in individual_list])
        elif measure == 'max_fitness':
            donor_idx = np.argmax([ind.fitness for ind in individual_list])
    
        donor = individual_list[donor_idx]
        recipients = individual_list[:donor_idx]
        recipients.extend(individual_list[donor_idx+1:])

        #choose at random a block from the donor
        if donor.size > 1:
            donation_idx = random.randint(1, donor.size - 1 )

            xo_recipients = ()

            for recipient in recipients:
                recipient_offs = Individual(collection=[*recipient.collection, donor[donation_idx]]
                         if reconstruct else None,
                          train_semantics=torch.stack([*recipient.train_semantics, donor.train_semantics[donation_idx]]),
                          test_semantics=torch.stack([*recipient.test_semantics, donor.test_semantics[donation_idx]])
                          if donor.test_semantics is not None
                          else None,
                          reconstruct=reconstruct)
                recipient_offs.nodes_collection = recipient.nodes_collection + [donor.nodes_collection[donation_idx]]
                recipient_offs.depth_collection = recipient.depth_collection + [donor.depth_collection[donation_idx]]
                recipient_offs.size = recipient.size + 1
                recipient_offs.nodes_count = sum(recipient_offs.nodes_collection) + (recipient_offs.size - 1)
                recipient_offs.depth = max([depth - (i - 1) if i != 0 else depth
                                for i, depth in enumerate(recipient_offs.depth_collection)]
                            ) + (recipient_offs.size - 1)

                xo_recipients += xo_recipients + (recipient_offs,)
                #xo_recipients.extend(recipient_offs)

            donor_offs = Individual(collection=donor.collection[:donation_idx] + donor.collection[donation_idx+1:]
                         if reconstruct else None,
                          train_semantics=torch.stack(
                                [*donor.train_semantics[:donation_idx], *donor.train_semantics[donation_idx + 1:]]),
                          test_semantics=torch.stack(
                [*donor.test_semantics[:donation_idx], *donor.test_semantics[donation_idx + 1:]])
                          if donor.test_semantics is not None
                          else None,
                          reconstruct=reconstruct)

            donor_offs.nodes_collection = donor.nodes_collection[:donation_idx] + donor.nodes_collection[donation_idx + 1:]
            donor_offs.depth_collection = donor.depth_collection[:donation_idx] + donor.depth_collection[donation_idx + 1:]
            donor_offs.size = donor.size - 1
            donor_offs.nodes_count = sum(donor_offs.nodes_collection) + (donor_offs.size - 1)
            donor_offs.depth = max([depth - (i - 1) if i != 0 else depth
                                for i, depth in enumerate(donor_offs.depth_collection)]
                            ) + (donor_offs.size - 1)

            #xo_recipients += (donor_offs,)
            xo_recipients = xo_recipients[:donation_idx] + (donor_offs,) + xo_recipients[donor_idx:]
            return *xo_recipients,

        else:
            return *individual_list,
    return best_d_n_xo

def new_donor_n_xo(n=5):
    """
    Generate a parallel donor_n_gxo function.

    Parameters
    ----------
    n: int
        Number of individuals to be used in crossover (1 donor, n-1 receivers).

    Returns
    -------
    Callable
        A donor_n_gxo crossover function (`paralel_d_n_xo`).
        Performs crossover between n individuals by donating a block of the donor to the n individuals in the receiver_list.

    Parameters
    ----------
    individual_list: list
        A list of Trees with n individuals to be used on donor_n_gxo.
    reconstruct : bool, optional
        Whether to reconstruct the Individual's collection after crossover (default: True).

    Returns
    -------
    Individuals
        The donor_n_gxo crossover tree n Individuals.
    """

    def crossover_for_recipient(donor, recipient, donation_idx, reconstruct):
        """Helper function to apply crossover on a single recipient."""
        recipient_offs = Individual(
            collection=[*recipient.collection, donor[donation_idx]] if reconstruct else None,
            train_semantics=torch.stack([*recipient.train_semantics, donor.train_semantics[donation_idx]]),
            test_semantics=torch.stack([*recipient.test_semantics, donor.test_semantics[donation_idx]])
            if donor.test_semantics is not None else None,
            reconstruct=reconstruct
        )

        recipient_offs.nodes_collection = recipient.nodes_collection + [donor.nodes_collection[donation_idx]]
        recipient_offs.depth_collection = recipient.depth_collection + [donor.depth_collection[donation_idx]]
        recipient_offs.size = recipient.size + 1
        recipient_offs.nodes_count = sum(recipient_offs.nodes_collection) + (recipient_offs.size - 1)
        recipient_offs.depth = max([depth - (i - 1) if i != 0 else depth
                                    for i, depth in enumerate(recipient_offs.depth_collection)]) + (
                                           recipient_offs.size - 1)

        return recipient_offs

    def new_d_n_xo(individual_list, reconstruct=True):
        # Choose at random the index of the donor (0 = p1, 1= p2)
        donor_idx = random.randint(0, n - 1)
        donor = individual_list[donor_idx]
        recipients = individual_list[:donor_idx] + individual_list[donor_idx + 1:]

        # Choose at random a block from the donor
        if donor.size > 1:
            donation_idx = random.randint(1, donor.size - 1)

            xo_recipients = list(crossover_for_recipient(donor, recipient, donation_idx, reconstruct) for recipient in recipients)

            # Apply crossover on the donor itself
            donor_offs = Individual(
                collection=donor.collection[:donation_idx] + donor.collection[
                                                             donation_idx + 1:] if reconstruct else None,
                train_semantics=torch.stack(
                    [*donor.train_semantics[:donation_idx], *donor.train_semantics[donation_idx + 1:]]),
                test_semantics=torch.stack(
                    [*donor.test_semantics[:donation_idx], *donor.test_semantics[donation_idx + 1:]])
                if donor.test_semantics is not None else None,
                reconstruct=reconstruct
            )

            donor_offs.nodes_collection = donor.nodes_collection[:donation_idx] + donor.nodes_collection[
                                                                                  donation_idx + 1:]
            donor_offs.depth_collection = donor.depth_collection[:donation_idx] + donor.depth_collection[
                                                                                  donation_idx + 1:]
            donor_offs.size = donor.size - 1
            donor_offs.nodes_count = sum(donor_offs.nodes_collection) + (donor_offs.size - 1)
            donor_offs.depth = max([depth - (i - 1) if i != 0 else depth
                                    for i, depth in enumerate(donor_offs.depth_collection)]) + (donor_offs.size - 1)

            # Return the donor and the crossover recipients
            xo_recipients = xo_recipients[:donation_idx] + [donor_offs] + xo_recipients[donor_idx:]
            return *xo_recipients,

        else:
            return *individual_list,

    return new_d_n_xo


def new_best_donor_n_xo(n=5, measure='min_fitness'):
    """
    Generate a best_donor_n_xo function.

    Parameters
    ----------
    n: int
        Number of individuals to be used in crossover (1 donor, n-1 recievers).
    measure: string
        Measure to be considered when selection the 'best' individual to be donor.

    Returns
    -------
    Callable
        An best_donor_n_xo crossover function (`best_d_n_xo`).

        Performs crossover between n individuals by donating a block of the donor to the n individuals in the reciever_list

        Parameters
        ----------
        individual_list: list
            A list of Trees with n individuals to be used on donor_n_gxo.
        reconstruct : bool, optional
            Whether to reconstruct the Individual's collection after crossover (default: True).

        Returns
        -------
        Individuals
            The best_donor_n_xo crossover tree n Individuals.
    """
    def crossover_for_recipient(donor, recipient, donation_idx, reconstruct):
        """Helper function to apply crossover on a single recipient."""
        recipient_offs = Individual(
            collection=[*recipient.collection, donor[donation_idx]] if reconstruct else None,
            train_semantics=torch.stack([*recipient.train_semantics, donor.train_semantics[donation_idx]]),
            test_semantics=torch.stack([*recipient.test_semantics, donor.test_semantics[donation_idx]])
            if donor.test_semantics is not None else None,
            reconstruct=reconstruct
        )

        recipient_offs.nodes_collection = recipient.nodes_collection + [donor.nodes_collection[donation_idx]]
        recipient_offs.depth_collection = recipient.depth_collection + [donor.depth_collection[donation_idx]]
        recipient_offs.size = recipient.size + 1
        recipient_offs.nodes_count = sum(recipient_offs.nodes_collection) + (recipient_offs.size - 1)
        recipient_offs.depth = max([depth - (i - 1) if i != 0 else depth
                                    for i, depth in enumerate(recipient_offs.depth_collection)]) + (
                                           recipient_offs.size - 1)

        return recipient_offs
    def new_best_d_n_xo(individual_list, reconstruct=True):

        _ = n  # dummy variable so n is identifiable

        # choose best individual to be donor:
        if measure == 'biggest':
            donor_idx = np.argmax([ind.size for ind in individual_list])
        elif measure == 'min_fitness':
            donor_idx = np.argmin([ind.fitness for ind in individual_list])
        elif measure == 'max_fitness':
            donor_idx = np.argmax([ind.fitness for ind in individual_list])

        donor = individual_list[donor_idx]
        recipients = individual_list[:donor_idx] + individual_list[donor_idx + 1:]

        # choose at random a block from the donor
        if donor.size > 1:
            donation_idx = random.randint(1, donor.size - 1)

            xo_recipients = list(
                crossover_for_recipient(donor, recipient, donation_idx, reconstruct) for recipient in recipients)

            donor_offs = Individual(collection=donor.collection[:donation_idx] + donor.collection[donation_idx + 1:]
            if reconstruct else None,
                                    train_semantics=torch.stack(
                                        [*donor.train_semantics[:donation_idx],
                                         *donor.train_semantics[donation_idx + 1:]]),
                                    test_semantics=torch.stack(
                                        [*donor.test_semantics[:donation_idx],
                                         *donor.test_semantics[donation_idx + 1:]])
                                    if donor.test_semantics is not None
                                    else None,
                                    reconstruct=reconstruct)

            donor_offs.nodes_collection = donor.nodes_collection[:donation_idx] + donor.nodes_collection[
                                                                                  donation_idx + 1:]
            donor_offs.depth_collection = donor.depth_collection[:donation_idx] + donor.depth_collection[
                                                                                  donation_idx + 1:]
            donor_offs.size = donor.size - 1
            donor_offs.nodes_count = sum(donor_offs.nodes_collection) + (donor_offs.size - 1)
            donor_offs.depth = max([depth - (i - 1) if i != 0 else depth
                                    for i, depth in enumerate(donor_offs.depth_collection)]
                                   ) + (donor_offs.size - 1)

            # xo_recipients += (donor_offs,)
            xo_recipients = xo_recipients[:donation_idx] + [donor_offs] + xo_recipients[donor_idx:]
            return *xo_recipients,

        else:
            return *individual_list,

    return new_best_d_n_xo

def dif_donor_n_xo(n=5):
    """
    Generate a parallel donor_n_gxo function.

    Parameters
    ----------
    n: int
        Number of individuals to be used in crossover (1 donor, n-1 receivers).

    Returns
    -------
    Callable
        A donor_n_gxo crossover function (`paralel_d_n_xo`).
        Performs crossover between n individuals by donating a block of the donor to the n individuals in the receiver_list.

    Parameters
    ----------
    individual_list: list
        A list of Trees with n individuals to be used on donor_n_gxo.
    reconstruct : bool, optional
        Whether to reconstruct the Individual's collection after crossover (default: True).

    Returns
    -------
    Individuals
        The donor_n_gxo crossover tree n Individuals.
    """

    def crossover_for_recipient(donor, recipient, donation_idx, reconstruct):
        """Helper function to apply crossover on a single recipient."""

        recipient_offs = Individual(
            collection=[*recipient.collection, donor[donation_idx]] if reconstruct else None,
            train_semantics=torch.stack([*recipient.train_semantics, donor.train_semantics[donation_idx]]),
            test_semantics=torch.stack([*recipient.test_semantics, donor.test_semantics[donation_idx]])
            if donor.test_semantics is not None else None,
            reconstruct=reconstruct
        )

        recipient_offs.nodes_collection = recipient.nodes_collection + [donor.nodes_collection[donation_idx]]
        recipient_offs.depth_collection = recipient.depth_collection + [donor.depth_collection[donation_idx]]
        recipient_offs.size = recipient.size + 1
        recipient_offs.nodes_count = sum(recipient_offs.nodes_collection) + (recipient_offs.size - 1)
        recipient_offs.depth = max([depth - (i - 1) if i != 0 else depth
                                    for i, depth in enumerate(recipient_offs.depth_collection)]) + (
                                           recipient_offs.size - 1)

        return recipient_offs

    def dif_d_n_xo(individual_list, reconstruct=True):
        # Choose at random the index of the donor (0 = p1, 1= p2)
        donor_idx = random.randint(0, n - 1)
        donor = individual_list[donor_idx]
        recipients = individual_list[:donor_idx] + individual_list[donor_idx + 1:]

        # Choose at random a block from the donor
        if donor.size > 1:
            #donation_idx = random.randint(1, donor.size - 1)

            xo_recipients = list(crossover_for_recipient(donor, recipient, random.randint(1, donor.size - 1), reconstruct) for recipient in recipients)

            # Return the donor and the crossover recipients
            xo_recipients = [donor] + xo_recipients
            return *xo_recipients,

        else:
            return *individual_list,

    return dif_d_n_xo

def dif_best_donor_n_xo(n=5, measure='min_fitness'):
    """
    Generate a best_donor_n_xo function.

    Parameters
    ----------
    n: int
        Number of individuals to be used in crossover (1 donor, n-1 recievers).
    measure: string
        Measure to be considered when selection the 'best' individual to be donor.

    Returns
    -------
    Callable
        An best_donor_n_xo crossover function (`best_d_n_xo`).

        Performs crossover between n individuals by donating a block of the donor to the n individuals in the reciever_list

        Parameters
        ----------
        individual_list: list
            A list of Trees with n individuals to be used on donor_n_gxo.
        reconstruct : bool, optional
            Whether to reconstruct the Individual's collection after crossover (default: True).

        Returns
        -------
        Individuals
            The best_donor_n_xo crossover tree n Individuals.
    """
    def crossover_for_recipient(donor, recipient, donation_idx, reconstruct):
        """Helper function to apply crossover on a single recipient."""
        recipient_offs = Individual(
            collection=[*recipient.collection, donor[donation_idx]] if reconstruct else None,
            train_semantics=torch.stack([*recipient.train_semantics, donor.train_semantics[donation_idx]]),
            test_semantics=torch.stack([*recipient.test_semantics, donor.test_semantics[donation_idx]])
            if donor.test_semantics is not None else None,
            reconstruct=reconstruct
        )

        recipient_offs.nodes_collection = recipient.nodes_collection + [donor.nodes_collection[donation_idx]]
        recipient_offs.depth_collection = recipient.depth_collection + [donor.depth_collection[donation_idx]]
        recipient_offs.size = recipient.size + 1
        recipient_offs.nodes_count = sum(recipient_offs.nodes_collection) + (recipient_offs.size - 1)
        recipient_offs.depth = max([depth - (i - 1) if i != 0 else depth
                                    for i, depth in enumerate(recipient_offs.depth_collection)]) + (
                                           recipient_offs.size - 1)

        return recipient_offs
    def dif_best_d_n_xo(individual_list, reconstruct=True):

        _ = n  # dummy variable so n is identifiable

        # choose best individual to be donor:
        if measure == 'biggest':
            donor_idx = np.argmax([ind.size for ind in individual_list])
        elif measure == 'min_fitness':
            donor_idx = np.argmin([ind.fitness for ind in individual_list])
        elif measure == 'max_fitness':
            donor_idx = np.argmax([ind.fitness for ind in individual_list])

        donor = individual_list[donor_idx]
        recipients = individual_list[:donor_idx] + individual_list[donor_idx + 1:]

        # choose at random a block from the donor
        if donor.size > 1:

            xo_recipients = list(
                crossover_for_recipient(donor, recipient, random.randint(1, donor.size - 1), reconstruct) for recipient in recipients)

            # xo_recipients += (donor_offs,)
            xo_recipients = [donor] + xo_recipients[donor_idx:]
            return *xo_recipients,

        else:
            return *individual_list,

    return dif_best_d_n_xo