import numpy as np
from typing import List, Tuple, Set, Dict, Optional
from dataclasses import dataclass
from enum import Enum
import copy

class Color(Enum):
    RED = 0
    BLUE = 1
    BLACK = 2
    ORANGE = 3
    
class TileType(Enum):
    NORMAL = 0
    JOKER = 1

@dataclass
class Tile:
    """Represents a single Rummikub tile"""
    color: Optional[Color]  # None for jokers
    number: Optional[int]   # None for jokers, 1-13 for normal tiles
    tile_type: TileType
    tile_id: int  # Unique identifier for each physical tile
    
    def __hash__(self):
        return hash(self.tile_id)
    
    def __eq__(self, other):
        if not isinstance(other, Tile):
            return False
        return self.tile_id == other.tile_id
    
    def __repr__(self):
        if self.tile_type == TileType.JOKER:
            return "JOKER"
        return f"{self.color.name[0]}{self.number}"
    
    def get_value(self) -> int:
        """Returns the point value of the tile"""
        if self.tile_type == TileType.JOKER:
            return 30  # Joker penalty
        return self.number

@dataclass
class TileSet:
    """Represents a set of tiles on the table (either a group or a run)"""
    tiles: List[Tile]
    set_type: str  # "group" or "run"
    
    def is_valid(self) -> bool:
        """Check if this set is valid according to Rummikub rules"""
        if len(self.tiles) < 3:
            return False
            
        if self.set_type == "group":
            return self._is_valid_group()
        elif self.set_type == "run":
            return self._is_valid_run()
        return False
    
    def _is_valid_group(self) -> bool:
        """Check if tiles form a valid group (3-4 same numbers, different colors)"""
        if len(self.tiles) < 3 or len(self.tiles) > 4:
            return False
        
        # Extract numbers (handling jokers)
        numbers = []
        colors = []
        joker_count = 0
        
        for tile in self.tiles:
            if tile.tile_type == TileType.JOKER:
                joker_count += 1
            else:
                numbers.append(tile.number)
                colors.append(tile.color)
        
        # All non-joker tiles must have the same number
        if len(set(numbers)) > 1:
            return False
        
        # All non-joker tiles must have different colors
        if len(colors) != len(set(colors)):
            return False
        
        return True
    
    def _is_valid_run(self) -> bool:
        """Check if tiles form a valid run (3+ consecutive numbers, same color)"""
        if len(self.tiles) < 3:
            return False
        
        # Extract colors and numbers (handling jokers)
        colors = []
        numbers = []
        joker_positions = []
        
        for i, tile in enumerate(self.tiles):
            if tile.tile_type == TileType.JOKER:
                joker_positions.append(i)
            else:
                colors.append(tile.color)
                numbers.append(tile.number)
        
        # All non-joker tiles must have the same color
        if len(set(colors)) > 1:
            return False
        
        # Check if numbers form a consecutive sequence
        if len(numbers) > 0:
            numbers.sort()
            # With jokers, check if we can fill gaps
            expected_length = len(self.tiles)
            min_num = numbers[0]
            max_num = numbers[-1]
            
            # Check if the range is valid
            if max_num - min_num + 1 > expected_length:
                return False
            
            # Check if numbers with jokers can form consecutive sequence
            all_positions = list(range(min_num, max_num + 1))
            missing_count = len(all_positions) - len(numbers)
            
            if missing_count != len(joker_positions):
                return False
        
        return True
    
    def get_value(self) -> int:
        """Returns the total value of tiles in this set"""
        return sum(tile.get_value() if tile.tile_type != TileType.JOKER 
                   else 0 for tile in self.tiles)


class RummikubAction:
    """Represents an action in Rummikub"""
    def __init__(self, action_type: str, tiles: List[Tile] = None, 
                 sets: List[TileSet] = None, table_config: List[TileSet] = None):
        """
        action_type: 'draw', 'initial_meld', 'play', 'end_turn'
        tiles: tiles from hand being played
        sets: new sets being formed
        table_config: complete table configuration after manipulation
        """
        self.action_type = action_type
        self.tiles = tiles or []
        self.sets = sets or []
        self.table_config = table_config


class RummikubEnv:
    """Rummikub Environment for Reinforcement Learning"""
    
    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.RandomState(seed)
        self.tiles_deck: List[Tile] = []
        self.player_hands: List[List[Tile]] = [[], []]  # 2 players
        self.table: List[TileSet] = []  # Sets on the table
        self.current_player: int = 0
        self.has_melded: List[bool] = [False, False]  # Track initial meld
        self.game_over: bool = False
        self.winner: Optional[int] = None
        self.turn_count: int = 0
        
        # For reward calculation
        self.previous_hand_values: List[int] = [0, 0]
        
        self._initialize_deck()
    
    def _initialize_deck(self):
        """Create the full deck of 106 tiles"""
        self.tiles_deck = []
        tile_id = 0
        
        # Create numbered tiles (2 copies of each color-number combination)
        for copy in range(2):
            for color in Color:
                for number in range(1, 14):
                    tile = Tile(color=color, number=number, 
                               tile_type=TileType.NORMAL, tile_id=tile_id)
                    self.tiles_deck.append(tile)
                    tile_id += 1
        
        # Create 2 jokers
        for _ in range(2):
            tile = Tile(color=None, number=None, 
                       tile_type=TileType.JOKER, tile_id=tile_id)
            self.tiles_deck.append(tile)
            tile_id += 1
    
    def reset(self) -> Dict:
        """Reset the environment for a new game"""
        # Shuffle the deck
        self.rng.shuffle(self.tiles_deck)
        
        # Deal 14 tiles to each player
        self.player_hands = [[], []]
        for player in range(2):
            self.player_hands[player] = self.tiles_deck[:14]
            self.tiles_deck = self.tiles_deck[14:]
        
        # Reset game state
        self.table = []
        self.current_player = 0
        self.has_melded = [False, False]
        self.game_over = False
        self.winner = None
        self.turn_count = 0
        
        # Initialize hand values for reward calculation
        self.previous_hand_values = [
            self._calculate_hand_value(0),
            self._calculate_hand_value(1)
        ]
        
        return self._get_state()
    
    def _calculate_hand_value(self, player: int) -> int:
        """Calculate total value of tiles in player's hand"""
        return sum(tile.get_value() for tile in self.player_hands[player])
    
    def _count_jokers_in_hand(self, player: int) -> int:
        """Count number of jokers in player's hand"""
        return sum(1 for tile in self.player_hands[player] 
                   if tile.tile_type == TileType.JOKER)
    
    def _get_state(self) -> Dict:
        """
        Return the current game state as defined by user:
        1. Board configuration and current player's hand
        2. Number of tiles opponent has
        3. Number of tiles to be drawn (pool size)
        """
        return {
            # Core state components
            'my_hand': copy.deepcopy(self.player_hands[self.current_player]),
            'table': copy.deepcopy(self.table),
            'opponent_tile_count': len(self.player_hands[1 - self.current_player]),
            'pool_size': len(self.tiles_deck),
            
            # Additional useful info
            'current_player': self.current_player,
            'has_melded': self.has_melded.copy(),
            'game_over': self.game_over,
            'winner': self.winner,
            'turn_count': self.turn_count
        }
    
    def get_legal_actions(self, player: int) -> List[RummikubAction]:
        """
        Get all legal actions for the current player.
        This is where you'd integrate with your ILP solver.
        
        For now, returns basic actions: draw or play valid sets
        """
        legal_actions = []
        
        # Option 1: Draw a tile (always legal if pool not empty)
        if len(self.tiles_deck) > 0:
            legal_actions.append(RummikubAction(action_type='draw'))
        
        # Option 2: Play tiles from hand
        if self.has_melded[player]:
            # Player has already made initial meld, can do any valid play
            # TODO: Integrate ILP solver here to find all valid plays
            legal_actions.extend(self._find_valid_plays(player))
        else:
            # Player needs to make initial meld (30+ points)
            legal_actions.extend(self._find_valid_initial_melds(player))
        
        return legal_actions
    
    def _find_valid_initial_melds(self, player: int) -> List[RummikubAction]:
        """
        Find all valid initial melds (sets that sum to 30+ points).
        This should call your ILP solver.
        """
        # Placeholder: basic implementation
        # TODO: Replace with ILP solver
        valid_melds = []
        hand = self.player_hands[player]
        
        # Try to find combinations that sum to >= 30
        # This is simplified - you'll want to use ILP solver
        for i in range(len(hand)):
            for j in range(i+1, len(hand)):
                for k in range(j+1, len(hand)):
                    test_set = TileSet(tiles=[hand[i], hand[j], hand[k]], 
                                      set_type='group')
                    if test_set.is_valid() and test_set.get_value() >= 30:
                        action = RummikubAction(
                            action_type='initial_meld',
                            tiles=[hand[i], hand[j], hand[k]],
                            sets=[test_set],
                            table_config=self.table + [test_set]
                        )
                        valid_melds.append(action)
        
        return valid_melds
    
    def _find_valid_plays(self, player: int) -> List[RummikubAction]:
        """
        Find all valid plays after initial meld.
        This should call your ILP solver for complex manipulations.
        """
        # Placeholder: basic implementation
        # TODO: Replace with ILP solver
        return []
    
    def step(self, action: RummikubAction) -> Tuple[Dict, float, bool, Dict]:
        """
        Execute an action and return (state, reward, done, info)
        
        Implements the user's reward function:
        1. R_t = (Sum of hand at t-1) - (Sum of hand at t)
        2. Win: R_T = 200 + sum of opponent's hand
        3. Lose: R_T = -sum of my hand
        4. Ice-breaking bonus: +20
        5. Drawing penalty: -5
        6. Joker penalty at end: -30 per joker
        """
        if self.game_over:
            raise ValueError("Game is already over")
        
        # Store hand value before action
        hand_value_before = self._calculate_hand_value(self.current_player)
        
        # Initialize info dictionary
        info = {
            'action_type': action.action_type,
            'tiles_played': 0,
            'drew_tile': False,
            'ice_broken': False,
            'joker_retrieved': False,
            'manipulation_occurred': False,
            'draw_penalty_applied': False,
            'invalid_action': False,
            'hand_size_before': len(self.player_hands[self.current_player]),
            'hand_value_before': hand_value_before,
        }
        
        reward = 0
        
        # Execute action
        if action.action_type == 'draw':
            # Draw a tile from the pool
            if len(self.tiles_deck) > 0:
                drawn_tile = self.tiles_deck.pop(0)
                self.player_hands[self.current_player].append(drawn_tile)
                info['drew_tile'] = True
                info['draw_penalty_applied'] = True
                reward -= 5  # Drawing penalty
            
        elif action.action_type == 'initial_meld':
            # Player makes initial meld
            if self._validate_initial_meld(action):
                self._apply_meld(action)
                self.has_melded[self.current_player] = True
                info['ice_broken'] = True
                info['tiles_played'] = len(action.tiles)
                reward += 20  # Ice-breaking bonus
            else:
                # Invalid meld - mark as invalid
                info['invalid_action'] = True
                # Don't apply penalty in reward, handle this separately if needed
                
        elif action.action_type == 'play':
            # Player plays tiles (after initial meld)
            if self._validate_play(action):
                info['tiles_played'] = len(action.tiles)
                # Check if manipulation occurred
                if len(action.table_config) != len(self.table) + len(action.sets):
                    info['manipulation_occurred'] = True
                self._apply_play(action)
            else:
                info['invalid_action'] = True
        
        # Calculate hand value after action
        hand_value_after = self._calculate_hand_value(self.current_player)
        info['hand_value_after'] = hand_value_after
        info['hand_size_after'] = len(self.player_hands[self.current_player])
        
        # Apply main reward: R_t = (hand value before) - (hand value after)
        if not info['invalid_action']:
            reward += (hand_value_before - hand_value_after)
        
        # Check termination conditions
        done = False
        
        # Condition 1: Current player has no tiles left (WIN)
        if len(self.player_hands[self.current_player]) == 0:
            self.game_over = True
            self.winner = self.current_player
            done = True
            
            # Calculate terminal reward for winner
            opponent = 1 - self.current_player
            opponent_hand_value = self._calculate_hand_value(opponent)
            reward = 200 + opponent_hand_value
            
            info['final_opponent_hand_value'] = opponent_hand_value
            info['win_type'] = 'emptied_hand'
            info['winner'] = self.current_player
        
        # Condition 2: No more tiles in pool and no possible moves
        elif len(self.tiles_deck) == 0:
            # Check if current player has any legal actions besides draw
            legal_actions = self.get_legal_actions(self.current_player)
            legal_actions = [a for a in legal_actions if a.action_type != 'draw']
            
            if len(legal_actions) == 0:
                # Game ends, determine winner by hand value
                self.game_over = True
                done = True
                
                player_value = self._calculate_hand_value(self.current_player)
                opponent_value = self._calculate_hand_value(1 - self.current_player)
                
                # Apply joker penalties
                player_jokers = self._count_jokers_in_hand(self.current_player)
                opponent_jokers = self._count_jokers_in_hand(1 - self.current_player)
                
                player_total = player_value  # Joker already worth 30 in calculation
                opponent_total = opponent_value
                
                info['jokers_in_hand'] = player_jokers
                
                if player_total < opponent_total:
                    # Current player wins
                    self.winner = self.current_player
                    reward = 200 + opponent_total
                    info['win_type'] = 'lowest_hand'
                    info['winner'] = self.current_player
                else:
                    # Current player loses
                    self.winner = 1 - self.current_player
                    reward = -player_total
                    info['win_type'] = 'lowest_hand'
                    info['winner'] = 1 - self.current_player
                
                info['final_my_hand_value'] = player_total
                info['final_opponent_hand_value'] = opponent_total
        
        # Update previous hand value for next turn
        self.previous_hand_values[self.current_player] = hand_value_after
        
        # Switch to next player
        if not done:
            self.current_player = 1 - self.current_player
            self.turn_count += 1
        
        state = self._get_state()
        return state, reward, done, info
    
    def _validate_initial_meld(self, action: RummikubAction) -> bool:
        """Validate that initial meld is legal (30+ points)"""
        if not action.sets:
            return False
        total_value = sum(s.get_value() for s in action.sets)
        all_valid = all(s.is_valid() for s in action.sets)
        # Check tiles come from player's hand
        all_tiles_in_hand = all(t in self.player_hands[self.current_player] 
                                for t in action.tiles)
        return total_value >= 30 and all_valid and all_tiles_in_hand
    
    def _validate_play(self, action: RummikubAction) -> bool:
        """Validate that a play is legal"""
        # Check that all resulting sets on table are valid
        if action.table_config is None:
            return False
        # Check tiles come from player's hand
        all_tiles_in_hand = all(t in self.player_hands[self.current_player] 
                                for t in action.tiles)
        return all(s.is_valid() for s in action.table_config) and all_tiles_in_hand
    
    def _apply_meld(self, action: RummikubAction):
        """Apply initial meld to game state"""
        # Remove tiles from hand
        for tile in action.tiles:
            self.player_hands[self.current_player].remove(tile)
        
        # Add sets to table
        self.table.extend(action.sets)
    
    def _apply_play(self, action: RummikubAction):
        """Apply a play to game state"""
        # Remove tiles from hand
        for tile in action.tiles:
            self.player_hands[self.current_player].remove(tile)
        
        # Update table with new configuration
        self.table = action.table_config
    
    def render(self):
        """Print the current game state"""
        print(f"\n{'='*60}")
        print(f"Turn {self.turn_count} - Player {self.current_player}'s turn")
        print(f"{'='*60}")
        
        for i, hand in enumerate(self.player_hands):
            value = self._calculate_hand_value(i)
            print(f"\nPlayer {i} hand ({len(hand)} tiles, value={value}): ", end="")
            if i == self.current_player:
                print([str(t) for t in hand])
            else:
                print(f"[{len(hand)} hidden tiles]")
        
        print(f"\nTable ({len(self.table)} sets):")
        for i, tile_set in enumerate(self.table):
            print(f"  Set {i+1} ({tile_set.set_type}): {[str(t) for t in tile_set.tiles]}")
        
        print(f"\nPool: {len(self.tiles_deck)} tiles remaining")
        print(f"Initial meld status: Player 0={self.has_melded[0]}, Player 1={self.has_melded[1]}")
        
        if self.game_over:
            print(f"\n{'='*60}")
            print(f"GAME OVER! Winner: Player {self.winner}")
            print(f"{'='*60}")


# Example usage
if __name__ == "__main__":
    env = RummikubEnv(seed=42)
    state = env.reset()
    
    print("Initial state:")
    env.render()
    
    # Example turn: draw a tile
    legal_actions = env.get_legal_actions(env.current_player)
    if legal_actions:
        action = legal_actions[0]  # Just draw for this example
        state, reward, done, info = env.step(action)
        print(f"\nAction taken: {action.action_type}")
        print(f"Reward: {reward}")
        print(f"Done: {done}")
        print(f"Info: {info}")
        env.render()