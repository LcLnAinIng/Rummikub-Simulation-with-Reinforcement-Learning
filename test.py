"""
Human vs ILP Opponent - Interactive Testing

This allows you to play Rummikub against the ILP baseline opponent
to test that the environment logic is working correctly.

Usage:
    python test_human_vs_opponent.py
"""

from Rummikub_env import RummikubEnv, RummikubAction, TileSet
from Baseline_Opponent import ILPOpponent
from Rummikub_ILP_Action_Generator import ActionGenerator, SolverMode


class HumanPlayer:
    """Interactive human player for testing."""
    
    def __init__(self):
        # Create action generator to show legal moves
        self.action_generator = ActionGenerator(mode=SolverMode.HYBRID, max_ilp_calls=30)
    
    def select_action(self, env: RummikubEnv) -> RummikubAction:
        """
        Let human select an action interactively.
        """
        current_player = env.current_player
        hand = env.player_hands[current_player]
        table = env.table
        has_melded = env.has_melded[current_player]
        pool_size = len(env.tiles_deck)
        
        print("\n" + "="*70)
        print("YOUR TURN")
        print("="*70)
        
        # Show game state
        self._display_state(env)
        
        # Get legal actions
        print("\nFinding legal actions...")
        legal_actions = self.action_generator.generate_all_legal_actions(
            hand, table, has_melded, pool_size
        )
        
        if len(legal_actions) == 0:
            print("ERROR: No legal actions found!")
            return None
        
        # Show legal actions
        print(f"\nYou have {len(legal_actions)} legal actions:")
        print()
        
        for i, action in enumerate(legal_actions):
            self._display_action(i, action)
        
        # Get user choice
        while True:
            try:
                choice = input(f"\nSelect action (0-{len(legal_actions)-1}): ").strip()
                idx = int(choice)
                
                if 0 <= idx < len(legal_actions):
                    return legal_actions[idx]
                else:
                    print(f"Invalid choice. Please enter 0-{len(legal_actions)-1}")
            except ValueError:
                print("Invalid input. Please enter a number.")
            except KeyboardInterrupt:
                print("\n\nGame interrupted by user.")
                return None
    
    def _display_state(self, env: RummikubEnv):
        """Display current game state."""
        current_player = env.current_player
        hand = env.player_hands[current_player]
        table = env.table
        opponent_hand_size = len(env.player_hands[1 - current_player])
        
        print(f"\nYour hand ({len(hand)} tiles, value={sum(t.get_value() for t in hand)}):")
        sorted_hand = sorted(hand, key=lambda t: (t.tile_type.value, 
                                                   t.color.value if t.color else -1, 
                                                   t.number if t.number else -1))
        for i, tile in enumerate(sorted_hand):
            print(f"  [{i}] {tile}", end="")
            if (i + 1) % 8 == 0:
                print()
        print()
        
        print(f"\nTable ({len(table)} sets):")
        if table:
            for i, tile_set in enumerate(table):
                tiles_str = ", ".join(str(t) for t in tile_set.tiles)
                value = sum(t.get_value() for t in tile_set.tiles if t.tile_type.name != 'JOKER')
                print(f"  Set {i+1} ({tile_set.set_type}, value={value}): [{tiles_str}]")
        else:
            print("  (empty)")
        
        print(f"\nOpponent: {opponent_hand_size} tiles")
        print(f"Pool: {len(env.tiles_deck)} tiles remaining")
        print(f"Has melded: {env.has_melded[current_player]}")
    
    def _display_action(self, idx: int, action: RummikubAction):
        """Display a single action option."""
        if action.action_type == 'draw':
            print(f"  [{idx}] DRAW a tile from pool")
        
        elif action.action_type == 'initial_meld':
            tiles_str = ", ".join(str(t) for t in action.tiles)
            total_value = sum(s.get_meld_value() for s in action.sets)
            print(f"  [{idx}] INITIAL MELD (value={total_value}):")
            print(f"      Play: {tiles_str}")
            for i, tile_set in enumerate(action.sets):
                set_tiles = ", ".join(str(t) for t in tile_set.tiles)
                print(f"      Set {i+1}: [{set_tiles}] ({tile_set.set_type})")
        
        elif action.action_type == 'play':
            tiles_str = ", ".join(str(t) for t in action.tiles)
            tiles_value = sum(t.get_value() for t in action.tiles)
            print(f"  [{idx}] PLAY {len(action.tiles)} tiles (value={tiles_value}):")
            print(f"      From hand: {tiles_str}")
            
            # Show if table changed
            if action.table_config:
                print(f"      Result: {len(action.table_config)} sets on table")


def play_game():
    """Main game loop for human vs opponent."""
    
    print("\n" + "="*70)
    print("RUMMIKUB: Human vs ILP Opponent")
    print("="*70)
    print("\nTesting the environment by playing against the baseline opponent.")
    print("This helps verify that all game logic is working correctly.")
    
    # Setup
    env = RummikubEnv(seed=None)  # Random seed for variety
    
    # Choose opponent type
    print("\nChoose opponent:")
    print("  [1] Model 1: Maximize value (simple greedy)")
    print("  [2] Model 2: Maximize value + minimize changes (smart)")
    
    while True:
        choice = input("Select opponent (1 or 2): ").strip()
        if choice == '1':
            opponent = ILPOpponent(objective='maximize_value')
            break
        elif choice == '2':
            opponent = ILPOpponent(objective='maximize_value_minimize_changes')
            break
        else:
            print("Invalid choice. Please enter 1 or 2.")
    
    human = HumanPlayer()
    
    # Choose player order
    print("\nChoose your position:")
    print("  [1] You go first")
    print("  [2] Opponent goes first")
    
    while True:
        choice = input("Select (1 or 2): ").strip()
        if choice == '1':
            human_player = 0
            break
        elif choice == '2':
            human_player = 1
            break
        else:
            print("Invalid choice. Please enter 1 or 2.")
    
    # Start game
    print("\n" + "="*70)
    print("GAME START!")
    print("="*70)
    
    state = env.reset()
    env.render()
    
    done = False
    turn_count = 0
    
    while not done:
        turn_count += 1
        print(f"\n{'='*70}")
        print(f"TURN {turn_count} - {'HUMAN' if env.current_player == human_player else 'OPPONENT'}")
        print(f"{'='*70}")
        
        if env.current_player == human_player:
            # Human's turn
            action = human.select_action(env)
            
            if action is None:
                print("Game interrupted.")
                return
            
            print(f"\nYou chose: {action.action_type}")
            
        else:
            # Opponent's turn
            print("\nOpponent is thinking...")
            
            action = opponent.select_action(
                env.player_hands[env.current_player],
                env.table,
                env.has_melded[env.current_player],
                len(env.tiles_deck)
            )
            
            print(f"Opponent chose: {action.action_type}")
            if action.action_type in ['play', 'initial_meld']:
                print(f"  Played {len(action.tiles)} tiles")
                tiles_str = ", ".join(str(t) for t in action.tiles)
                print(f"  Tiles: {tiles_str}")
        
        # Execute action
        state, reward, done, info = env.step(action)
        
        # Show result
        print(f"\nReward: {reward}")
        if info.get('ice_broken'):
            print("  🎉 Ice broken! (30+ points played)")
        if info.get('manipulation_occurred'):
            print("  🔄 Table manipulation occurred")
        
        # Show updated state
        env.render()
        
        # Check if game over
        if done:
            print("\n" + "="*70)
            print("GAME OVER!")
            print("="*70)
            
            if env.winner == human_player:
                print("\n🎉 YOU WIN! 🎉")
            elif env.winner == 1 - human_player:
                print("\n😞 OPPONENT WINS 😞")
            else:
                print("\n🤝 TIE 🤝")
            
            print(f"\nFinal scores:")
            print(f"  Your hand value: {sum(t.get_value() for t in env.player_hands[human_player])}")
            print(f"  Opponent hand value: {sum(t.get_value() for t in env.player_hands[1-human_player])}")
            print(f"\nTotal turns: {turn_count}")
            
            break
        
        # Pause between turns
        if env.current_player != human_player:
            input("\nPress Enter to continue...")


def main():
    """Main entry point."""
    while True:
        try:
            play_game()
            
            # Play again?
            print("\n" + "="*70)
            choice = input("\nPlay again? (y/n): ").strip().lower()
            if choice != 'y':
                print("\nThanks for testing! Goodbye!")
                break
                
        except KeyboardInterrupt:
            print("\n\nGame interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n\nERROR: {e}")
            import traceback
            traceback.print_exc()
            break


if __name__ == "__main__":
    main()