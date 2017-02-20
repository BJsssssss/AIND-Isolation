"""This file contains all the classes you must complete for this project.

You can use the test cases in agent_test.py to help during development, and
augment the test suite with your own test cases to further test your code.

You must test your agent's strength against a set of agents with known
relative strength using tournament.py and include the results in your report.
"""
import random


class Timeout(Exception):
    """Subclass base exception for code clarity."""
    pass


def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """

    if game.is_winner(player):
       return(float('inf'))
    elif game.is_loser(player):
       return(float('-inf'))
    else:
       self_moves_left = len(game.get_legal_moves(player))
       oppo_moves_left = len(game.get_legal_moves(game.get_opponent(player)))

       # More aggressive evaluation function set to the remaining difference between own and twice number of
       # opponent's remaining moves is used to increase pressure on opponent when LESS valid moves available
       # in a state. The multiplier 2 is arbitrary. It can be shown that any multiplier > 1 applied to opponent's move
       # number will produce similarly moving scores. So same results are expected for example multiplier 3, 4 or any
       # real multiplier n, for n > 1

       #return (float(self_moves_left - 2 * oppo_moves_left )) # Use any multipler n for opponent moves as  long as n > 1
       return (float(- oppo_moves_left )) # Use any multipler n for opponent moves as  long as n > 1

def eval_less_aggressive(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """

    if game.is_winner(player):
       return(float('inf'))
    elif game.is_loser(player):
       return(float('-inf'))
    else:
       self_moves_left = len(game.get_legal_moves(player))
       oppo_moves_left = len(game.get_legal_moves(game.get_opponent(player)))

       # less aggressive evaluation score for heuristic used - the remaining difference between own and only half number of
       # opponent's remaining moves. The multiplier 0.5 is arbitrary. It can be shown that any multiplier < 1 applied to
       # opponent's move number will produce similarly moving scores. So same results are expected for any multiplier
       # such as 0.25, 0.5, etc or any real multiplier n, for n < 1

       return (float(self_moves_left - 0.5 * oppo_moves_left))   # Use any multipler n for opponent moves as long as n < 1


def eval_less_oppo_moves(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """

    if game.is_winner(player):
       return(float('inf'))
    elif game.is_loser(player):
       return(float('-inf'))
    else:
       self_moves_left = len(game.get_legal_moves(player))
       oppo_moves_left = len(game.get_legal_moves(game.get_opponent(player)))

       # the evaluation score for heuristic is based on less number of opponent moves available in a game state
       # if there are less moves for the opponent, the score is higher. Own moves are not taken into account
       # as in more aggressive play used in custom score above to check the performance of this eval vs that one
       # Blank spaces do not reflect directly on L-moves possible since manoeuvering is possible even in congested
       # spaces as long as an L-move is possible. Blank spaces would have been a good input in a queen move game

       return (float(- oppo_moves_left))


class CustomPlayer:
    """Game-playing agent that chooses a move using your evaluation function
    and a depth-limited minimax algorithm with alpha-beta pruning. You must
    finish and test this player to make sure it properly uses minimax and
    alpha-beta to return a good move before the search time limit expires.

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    iterative : boolean (optional)
        Flag indicating whether to perform fixed-depth search (False) or
        iterative deepening search (True).

    method : {'minimax', 'alphabeta'} (optional)
        The name of the search method to use in get_move().

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """

    def __init__(self, search_depth=3, score_fn=custom_score,
                 iterative=True, method='minimax', timeout=10.):
        self.search_depth = search_depth
        self.iterative = iterative
        self.score = score_fn
        self.method = method
        self.time_left = None
        self.TIMER_THRESHOLD = timeout

    def get_move(self, game, legal_moves, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        This function must perform iterative deepening if self.iterative=True,
        and it must use the search method (minimax or alphabeta) corresponding
        to the self.method value.

        **********************************************************************
        NOTE: If time_left < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        legal_moves : list<(int, int)>
            A list containing legal moves. Moves are encoded as tuples of pairs
            of ints defining the next (row, col) for the agent to occupy.

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """

        self.time_left = time_left

        if self.time_left() < self.TIMER_THRESHOLD :
            raise Timeout()

        legal_moves = game.get_legal_moves(game.active_player)
        # check initial legal moves passed if valid
        if not legal_moves:
            return(-1,-1)  # Right away
        else:
            best_move = legal_moves[0] # Initialize best_move to atleast first element of list of moves

            try:
                # The search method call (alpha beta or minimax) should happen in
                # here in order to avoid timeout. The try/except block will
                # automatically catch the exception raised by the search method
                # when the timer gets close to expiring

                if self.time_left() < self.TIMER_THRESHOLD:  # Not sufficient time
                    return best_move

                if self.iterative:   #iterative deepening flag set
                    depth = 1 # Initialize depth

                    while True: # limiting depth not recommended, evaluate indefinitely till out of time, check time left below
                         if self.time_left() < self.TIMER_THRESHOLD:  # Not sufficient time
                             return best_move

                         if self.method == 'minimax':
                             _, move = self.minimax(game,depth)
                         else : # self.method == 'alphabeta'
                             _, move = self.alphabeta(game, depth)

                         if move != (-1,-1):  # recursive search did complete in time
                             best_move = move   # so far calculated as deep as possible
                         else:
                             break # decamp with last best move

                         depth +=1  #Increment depth and continue deeper

                else: # if ID not set
                     if self.method == 'minimax':
                         _, best_move = self.minimax(game,self.search_depth)
                     else: # select alphabeta
                         _, best_move = self.alphabeta(game, self.search_depth)


            except Timeout:
                # Handle any actions required at timeout, if necessary
                return best_move


        # Return the best move from the last completed search iteration
        return best_move


    def minimax(self, game, depth, maximizing_player=True):
        """Implement the minimax search algorithm as described in the lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """

        if self.time_left() < self.TIMER_THRESHOLD :
            raise Timeout()

        legal_moves = game.get_legal_moves(game.active_player)
        if not legal_moves:
           return self.score(game,self), (-1,-1)

        # Initialize best score and best move first
        if maximizing_player:
            best_score = float('-inf')
        else:
            best_score = float('inf')

        best_move = (-1, -1)  # Initialize best_move here to illegal as this will be used as a check in get_moves
                              # to check that recursive calls at longer depths have completed to depth 1

        if depth == 1:
           for valid_move in legal_moves:
               # Check if time left to run minmax again
               if self.time_left() < self.TIMER_THRESHOLD :  # Not sufficient time
                   return best_score, best_move

               if maximizing_player: # own player to maximize
                  score = self.score(game.forecast_move(valid_move), self)
                  if score > best_score:
                    best_score = score
                    best_move = valid_move

               else : #Opposing player to minimize
                   score = self.score(game.forecast_move(valid_move), self)
                   if score < best_score:
                      best_score = score
                      best_move = valid_move

           return best_score, best_move

        if depth > 1 :
           for valid_move in legal_moves:
           # Check if time left to run minmax again
               if self.time_left() < self.TIMER_THRESHOLD :  # Not sufficient time
                   return best_score, best_move
               #Otherwise run minmax recursively to next depth
               if maximizing_player: #own player to maximize
                  score, _ = self.minimax(game.forecast_move(valid_move), depth - 1, False)
                  if score > best_score:
                     best_score = score
                     best_move = valid_move
               else : #Opposing player to minimize
                  score, _ = self.minimax(game.forecast_move(valid_move), depth - 1, True)
                  if score < best_score:
                     best_score = score
                     best_move = valid_move

        return best_score, best_move



    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf"), maximizing_player=True):
        """Implement minimax search with alpha-beta pruning as described in the
        lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """

        if self.time_left() < self.TIMER_THRESHOLD :
            raise Timeout()

        legal_moves = game.get_legal_moves(game.active_player)
        if not legal_moves:
           return self.score(game,self), (-1,-1)

        # Initialize best score and best move first
        best_move = (-1, -1)  # Initialize best_move here to illegal as this will be used as a check in get_moves
                              # to check that recursive calls at longer depths have completed to depth 1

        if maximizing_player:
            best_score = float('-inf')
        else:
            best_score = float('inf')

        if depth == 1:
           for valid_move in legal_moves:
               #Check if time left to run minmax again
               if self.time_left() < self.TIMER_THRESHOLD :   # Not sufficient time
                   return best_score, best_move
               #Otherwise run minimax recursively to next depth

               if maximizing_player: #own player to maximize
                  score = self.score(game.forecast_move(valid_move), self)
                  if score > best_score:
                      best_score = score
                      best_move = valid_move
                      alpha = max(alpha, score)
                      if beta <= alpha :
                         break

               else : #Opposing player to minimize
                   score = self.score(game.forecast_move(valid_move), self)
                   if score < best_score:
                       best_score = score
                       best_move = valid_move
                       beta = min(beta, score)
                       if beta <= alpha :
                           break

           return best_score, best_move

        if depth > 1:

           for valid_move in legal_moves:
               # Check if time left to run minmax again
               if self.time_left() < self.TIMER_THRESHOLD :  # Not sufficient time
                   return best_score, best_move
               #Otherwise run alphabeta recursively to next depth

               if maximizing_player: #own player to maximize
                  score, _ = self.alphabeta(game.forecast_move(valid_move), depth - 1, alpha, beta,
                                            False)
                  if score > best_score:
                      best_score = score
                      best_move = valid_move
                      alpha = max(alpha, score)
                      if beta <= alpha :
                          break

               else : #Opposing player to minimize
                   score, _ = self.alphabeta(game.forecast_move(valid_move), depth - 1, alpha, beta,
                                             True)
                   if score < best_score:
                       best_score = score
                       best_move = valid_move
                       beta = min(beta, score)
                       if beta <= alpha :
                           break

        return best_score, best_move
