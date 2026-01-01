import numpy as np
from typing import List, Tuple, Set, Dict, Optional
from itertools import combinations, product
from dataclasses import dataclass
import copy
from Rummikub_env import Tile, TileSet, Color, TileType, RummikubAction


class ILPActionGenerator:
    """
    Generates ALL legal actions for Rummikub using ILP-inspired techniques.
    
    Strategy:
    1. Generate candidate actions heuristically
    2. Validate each action is legal
    3. Return all valid actions for RL agent to choose from
    """
    
    def __init__(self):
        # Pre-compute all possible valid sets (1174 total from paper)
        self.all_possible_sets = self._generate_all_possible_sets()
    
    def _generate_all_possible_sets(self) -> List[Dict]:
        """
        Pre-compute all 1174 possible valid sets as described in the paper.
        Each set is represented as a dict with tile requirements.
        
        Returns list of dicts: {
            'type': 'run' or 'group',
            'tiles_needed': list of (color, number) tuples,
            'jokers': number of jokers needed
        }
        """
        possible_sets = []
        
        # Runs without jokers (3, 4, 5 consecutive)
        for color in range(4):  # 4 colors
            for start in range(1, 14):  # numbers 1-13
                for length in [3, 4, 5]:
                    if start + length - 1 <= 13:
                        tiles = [(color, start + i) for i in range(length)]
                        possible_sets.append({
                            'type': 'run',
                            'tiles_needed': tiles,
                            'jokers': 0
                        })
        
        # Groups without jokers (3 or 4 same number, different colors)
        for number in range(1, 14):
            # 3 different colors
            for color_combo in combinations(range(4), 3):
                tiles = [(color, number) for color in color_combo]
                possible_sets.append({
                    'type': 'group',
                    'tiles_needed': tiles,
                    'jokers': 0
                })
            # 4 different colors
            tiles = [(color, number) for color in range(4)]
            possible_sets.append({
                'type': 'group',
                'tiles_needed': tiles,
                'jokers': 0
            })
        
        # TODO: Add sets with 1 joker and 2 jokers
        # This would add the remaining ~989 possible sets
        
        return possible_sets
    
    def generate_all_legal_actions(self, 
                                   hand_tiles: List[Tile],
                                   table_sets: List[TileSet],
                                   has_melded: bool,
                                   pool_size: int) -> List[RummikubAction]:
        """
        Generate ALL legal actions for the current game state.
        
        Args:
            hand_tiles: Tiles in player's hand
            table_sets: Current sets on the table
            has_melded: Whether player has made initial meld
            pool_size: Number of tiles left in pool
            
        Returns:
            List of all legal RummikubAction objects
        """
        legal_actions = []
        
        # Action 1: Draw a tile (always legal if pool not empty)
        if pool_size > 0:
            legal_actions.append(RummikubAction(action_type='draw'))
        
        if has_melded:
            # After initial meld: generate all possible plays
            legal_actions.extend(
                self._generate_post_meld_actions(hand_tiles, table_sets)
            )
        else:
            # Before initial meld: only actions >= 30 points
            legal_actions.extend(
                self._generate_initial_meld_actions(hand_tiles)
            )
        
        return legal_actions
    
    def _generate_initial_meld_actions(self, 
                                       hand_tiles: List[Tile]) -> List[RummikubAction]:
        """
        Generate all initial meld actions (must sum to >= 30 points).
        Only uses tiles from hand, no table manipulation.
        """
        legal_actions = []
        
        # Try all possible combinations of hand tiles
        # Start with larger subsets (more likely to reach 30 points)
        for size in range(len(hand_tiles), 2, -1):  # Down to 3 tiles minimum
            for tile_combo in combinations(hand_tiles, size):
                # Try to form valid sets from this combination
                possible_sets = self._find_valid_sets_from_tiles(list(tile_combo))
                
                for set_combination in possible_sets:
                    # Check if total value >= 30
                    total_value = sum(s.get_value() for s in set_combination)
                    if total_value >= 30:
                        # Get tiles used in these sets
                        tiles_used = []
                        for tile_set in set_combination:
                            tiles_used.extend(tile_set.tiles)
                        
                        action = RummikubAction(
                            action_type='initial_meld',
                            tiles=tiles_used,
                            sets=set_combination,
                            table_config=set_combination  # New table state
                        )
                        legal_actions.append(action)
        
        return legal_actions
    
    def _generate_post_meld_actions(self,
                                    hand_tiles: List[Tile],
                                    table_sets: List[TileSet]) -> List[RummikubAction]:
        """
        Generate all legal actions after initial meld.
        This includes:
        1. Playing new sets from hand
        2. Adding tiles to existing sets
        3. Manipulating table (splitting, combining sets)
        4. Complex manipulations combining hand + table
        """
        legal_actions = []
        
        # Strategy 1: Play complete sets from hand only
        legal_actions.extend(
            self._generate_hand_only_actions(hand_tiles, table_sets)
        )
        
        # Strategy 2: Add single tiles to existing table sets
        legal_actions.extend(
            self._generate_single_tile_additions(hand_tiles, table_sets)
        )
        
        # Strategy 3: Simple table manipulations + hand tiles
        legal_actions.extend(
            self._generate_simple_manipulations(hand_tiles, table_sets)
        )
        
        # Strategy 4: Complex manipulations using ILP
        # This is computationally expensive, so we might limit it
        legal_actions.extend(
            self._generate_complex_manipulations(hand_tiles, table_sets)
        )
        
        return legal_actions
    
    def _generate_hand_only_actions(self,
                                    hand_tiles: List[Tile],
                                    table_sets: List[TileSet]) -> List[RummikubAction]:
        """Generate actions that only play tiles from hand without table manipulation."""
        legal_actions = []
        
        # Try all subsets of hand tiles
        for size in range(3, len(hand_tiles) + 1):
            for tile_combo in combinations(hand_tiles, size):
                # Find valid sets from these tiles
                possible_sets = self._find_valid_sets_from_tiles(list(tile_combo))
                
                for set_combination in possible_sets:
                    tiles_used = []
                    for tile_set in set_combination:
                        tiles_used.extend(tile_set.tiles)
                    
                    # New table = old table + new sets
                    new_table = copy.deepcopy(table_sets) + set_combination
                    
                    action = RummikubAction(
                        action_type='play',
                        tiles=tiles_used,
                        sets=set_combination,
                        table_config=new_table
                    )
                    legal_actions.append(action)
        
        return legal_actions
    
    def _generate_single_tile_additions(self,
                                        hand_tiles: List[Tile],
                                        table_sets: List[TileSet]) -> List[RummikubAction]:
        """
        Generate actions that add single tiles from hand to existing table sets.
        Example: Hand has B6, table has {B7, B8, B9} → can add B6 to make {B6, B7, B8, B9}
        """
        legal_actions = []
        
        for tile in hand_tiles:
            for set_idx, table_set in enumerate(table_sets):
                # Try adding this tile to this set
                new_tiles = table_set.tiles + [tile]
                
                # Check if it forms a valid run
                test_run = TileSet(tiles=new_tiles, set_type='run')
                if test_run.is_valid():
                    new_table = copy.deepcopy(table_sets)
                    new_table[set_idx] = test_run
                    
                    action = RummikubAction(
                        action_type='play',
                        tiles=[tile],
                        sets=[test_run],
                        table_config=new_table
                    )
                    legal_actions.append(action)
                
                # Check if it forms a valid group
                test_group = TileSet(tiles=new_tiles, set_type='group')
                if test_group.is_valid():
                    new_table = copy.deepcopy(table_sets)
                    new_table[set_idx] = test_group
                    
                    action = RummikubAction(
                        action_type='play',
                        tiles=[tile],
                        sets=[test_group],
                        table_config=new_table
                    )
                    legal_actions.append(action)
        
        return legal_actions
    
    def _generate_simple_manipulations(self,
                                       hand_tiles: List[Tile],
                                       table_sets: List[TileSet]) -> List[RummikubAction]:
        """
        Generate simple table manipulations:
        1. Split a table set and use hand tiles to form new sets
        2. Take tiles from table sets and recombine with hand tiles
        """
        legal_actions = []
        
        # Try splitting each table set
        for set_idx, table_set in enumerate(table_sets):
            if table_set.set_type == 'run' and len(table_set.tiles) >= 4:
                # Try splitting the run at different positions
                for split_pos in range(1, len(table_set.tiles)):
                    left_tiles = table_set.tiles[:split_pos]
                    right_tiles = table_set.tiles[split_pos:]
                    
                    # Check if both parts are valid + hand tiles form valid sets
                    if len(left_tiles) >= 3 and len(right_tiles) >= 3:
                        # Try combining hand tiles with split parts
                        for hand_subset in self._get_hand_subsets(hand_tiles):
                            all_tiles = left_tiles + right_tiles + hand_subset
                            possible_sets = self._find_valid_sets_from_tiles(all_tiles)
                            
                            for set_combination in possible_sets:
                                # Build new table
                                new_table = [s for i, s in enumerate(table_sets) if i != set_idx]
                                new_table.extend(set_combination)
                                
                                action = RummikubAction(
                                    action_type='play',
                                    tiles=hand_subset,
                                    sets=set_combination,
                                    table_config=new_table
                                )
                                legal_actions.append(action)
        
        return legal_actions
    
    def _generate_complex_manipulations(self,
                                        hand_tiles: List[Tile],
                                        table_sets: List[TileSet]) -> List[RummikubAction]:
        """
        Use ILP to find complex manipulations.
        This is the most powerful but also most expensive method.
        
        Approach:
        1. For each subset of hand tiles to play
        2. Use ILP to check if they can be combined with table tiles
        3. If feasible, extract the solution as an action
        """
        legal_actions = []
        
        # This would call actual ILP solver
        # For now, placeholder
        
        # TODO: Implement full ILP solver as described in the paper
        # This requires:
        # - Setting up the constraint matrix (s_ij)
        # - Solving the ILP
        # - Extracting the solution to get table configuration
        
        return legal_actions
    
    def _find_valid_sets_from_tiles(self, 
                                    tiles: List[Tile]) -> List[List[TileSet]]:
        """
        Find all possible ways to partition tiles into valid sets.
        
        This is a recursive partition problem.
        Returns a list of partitions, where each partition is a list of TileSets.
        """
        if len(tiles) < 3:
            return []
        
        valid_partitions = []
        
        # Try to form one set and recursively partition the rest
        for size in range(3, min(len(tiles) + 1, 14)):  # Max set size
            for tile_combo in combinations(tiles, size):
                # Try as run
                test_run = TileSet(tiles=list(tile_combo), set_type='run')
                if test_run.is_valid():
                    remaining = [t for t in tiles if t not in tile_combo]
                    if len(remaining) == 0:
                        valid_partitions.append([test_run])
                    else:
                        sub_partitions = self._find_valid_sets_from_tiles(remaining)
                        for sub in sub_partitions:
                            valid_partitions.append([test_run] + sub)
                
                # Try as group
                test_group = TileSet(tiles=list(tile_combo), set_type='group')
                if test_group.is_valid():
                    remaining = [t for t in tiles if t not in tile_combo]
                    if len(remaining) == 0:
                        valid_partitions.append([test_group])
                    else:
                        sub_partitions = self._find_valid_sets_from_tiles(remaining)
                        for sub in sub_partitions:
                            valid_partitions.append([test_group] + sub)
        
        return valid_partitions
    
    def _get_hand_subsets(self, hand_tiles: List[Tile]) -> List[List[Tile]]:
        """Generate all subsets of hand tiles (for manipulation attempts)."""
        subsets = []
        for size in range(1, min(len(hand_tiles) + 1, 6)):  # Limit to 5 tiles
            for combo in combinations(hand_tiles, size):
                subsets.append(list(combo))
        return subsets
    
    # =====================================================
    # ILP SOLVER INTEGRATION (from the paper)
    # =====================================================
    
    def solve_ilp_optimal(self,
                          hand_tiles: List[Tile],
                          table_sets: List[TileSet]) -> Optional[RummikubAction]:
        """
        Use ILP to find the OPTIMAL action (maximize tiles/value played).
        This implements the paper's approach directly.
        
        Variables:
        - x_j: set j can be placed 0, 1, or 2 times
        - y_i: tile i can be placed 0, 1, or 2 from rack to table
        
        Constraints:
        - sum(s_ij * x_j) = t_i + y_i  (tile usage)
        - y_i <= r_i  (can't play more than you have)
        
        Objective:
        - Maximize sum(y_i) or sum(v_i * y_i)
        """
        # TODO: Implement using scipy.optimize.linprog or pulp
        # For now, return None
        return None


# ========================================================
# INTEGRATION WITH ENVIRONMENT
# ========================================================

def integrate_ilp_with_env():
    """
    Example of how to integrate ILP action generator with the environment.
    """
    
    pass


# ========================================================
# NOTES FOR IMPLEMENTATION
# ========================================================

"""
IMPLEMENTATION ROADMAP:

1. START SIMPLE:
   - Implement _find_valid_sets_from_tiles properly
   - Test with hand-only actions first
   - Gradually add complexity

2. OPTIMIZE PERFORMANCE:
   - Cache repeated calculations
   - Prune obviously bad actions early
   - Limit search depth for manipulations

3. FULL ILP INTEGRATION:
   - Implement solve_ilp_optimal using:
     * scipy.optimize (for simple cases)
     * pulp or cvxpy (for better control)
     * Gurobi/CPLEX (for best performance)
   
4. ACTION SPACE SIZE:
   - Initial meld: ~10-100 actions
   - Post-meld: ~100-1000 actions (with manipulations)
   - May need to sample/prune for tractable RL

5. TESTING:
   - Unit tests for each action generation method
   - Compare with paper's examples
   - Verify all actions are truly legal
"""