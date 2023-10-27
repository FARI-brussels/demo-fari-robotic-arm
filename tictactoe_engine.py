"""
Receive the real world positions of grid, x and o and return the position and letter of the next move
"""
import random
from flask import Flask, request, jsonify
import json

app = Flask(__name__)

GRID_SIZE = None

def infer_tic_tac_toe_state(bounding_boxes):
    # Extract the bounding box of the grid
    grid_center_x, grid_center_y, grid_w, grid_h = bounding_boxes['grid'][0]
    cell_w = grid_w / 3
    cell_h = grid_h / 3
    GRID_SIZE = cell_w, cell_h
    # Define the boundaries for the grid cells
    left_boundary = grid_center_x - (1.5 * cell_w)
    top_boundary = grid_center_y - (1.5 * cell_h)

    # Define a function to get the cell position given a point
    def get_cell(position):
        col = int((position[0] - left_boundary) // cell_w)
        row = int((position[1] - top_boundary) // cell_h)
        return row, col

    # Initialize an empty 3x3 grid
    grid_state = [[' ' for _ in range(3)] for _ in range(3)]
    
    # Fill in the 'X' marks
    for position in bounding_boxes.get('X', []):
        row, col = get_cell(position[:2])
        grid_state[row][col] = 'X'
    
    # Fill in the 'O' marks
    for position in bounding_boxes.get('O', []):
        row, col = get_cell(position[:2])
        grid_state[row][col] = 'O'

    return grid_state




def evaluate(board):
    # Check rows, columns, and diagonals for a win
    for i in range(3):
        if board[i][0] == board[i][1] == board[i][2] and board[i][0] != ' ':
            return 10 if board[i][0] == 'X' else -10
        if board[0][i] == board[1][i] == board[2][i] and board[0][i] != ' ':
            return 10 if board[0][i] == 'X' else -10

    if board[0][0] == board[1][1] == board[2][2] and board[0][0] != ' ':
        return 10 if board[0][0] == 'X' else -10

    if board[0][2] == board[1][1] == board[2][0] and board[0][2] != ' ':
        return 10 if board[0][2] == 'X' else -10

    return 0

def minimax(board, depth, is_max, alpha, beta, player_letter):
    score = evaluate(board)

    if score == 10:
        return score - depth

    if score == -10:
        return score + depth

    if not any(' ' in row for row in board):
        return 0

    if is_max:
        max_eval = float('-inf')
        for i in range(3):
            for j in range(3):
                if board[i][j] == ' ':
                    board[i][j] = 'X' if player_letter == 'X' else 'O'
                    value = minimax(board, depth+1, not is_max, alpha, beta, player_letter)
                    max_eval = max(max_eval, value)
                    alpha = max(alpha, value)
                    board[i][j] = ' '
                    if beta <= alpha:
                        break
        return max_eval

    else:
        min_eval = float('inf')
        for i in range(3):
            for j in range(3):
                if board[i][j] == ' ':
                    board[i][j] = 'O' if player_letter == 'X' else 'X'
                    value = minimax(board, depth+1, not is_max, alpha, beta, player_letter)
                    min_eval = min(min_eval, value)
                    beta = min(beta, value)
                    board[i][j] = ' '
                    if beta <= alpha:
                        break
        return min_eval

def find_best_move(board):
    x_count = sum(row.count('X') for row in board)
    o_count = sum(row.count('O') for row in board)
    
    if x_count > o_count:
        player_letter = 'O'
    elif x_count < o_count:
        player_letter = 'X'
    else:
        player_letter = random.choice(['X', 'O'])

    best_move = (-1, -1)
    best_val = float('-inf') if player_letter == 'X' else float('inf')
    
    for i in range(3):
        for j in range(3):
            if board[i][j] == ' ':
                board[i][j] = player_letter
                move_val = minimax(board, 0, player_letter == 'O', float('-inf'), float('inf'), player_letter)
                board[i][j] = ' '

                if player_letter == 'X' and move_val > best_val:
                    best_move = (i, j)
                    best_val = move_val

                if player_letter == 'O' and move_val < best_val:
                    best_move = (i, j)
                    best_val = move_val
    win, winner_letter = check_win(board)
    if win:
        return None, winner_letter
    return best_move, player_letter



def get_cell_center(index, grid_bbox):
    """
    Get the world coordinates of the center of the grid cell.

    Parameters:
    - index: tuple of row, column (0-indexed) of the cell.
    - grid_bbox: bounding box of the grid in the format (center_x, center_y, width, height)

    Returns:
    tuple (x, y) representing the world coordinates of the center of the cell.
    """

    row, col = index
    center_x, center_y, width, height = grid_bbox

    cell_width = width / 3
    cell_height = height / 3

    x = center_x - width / 2 + (col + 0.5) * cell_width
    y = center_y - height / 2 + (row + 0.5) * cell_height

    return (x, y)


def check_win(board):
    # Check rows, columns, and diagonals for a win
    for i in range(3):
        # Check rows
        if board[i][0] == board[i][1] == board[i][2] and board[i][0] != ' ':
            return True, board[i][0]

        # Check columns
        if board[0][i] == board[1][i] == board[2][i] and board[0][i] != ' ':
            return True, board[0][i]

    # Check diagonals
    if board[0][0] == board[1][1] == board[2][2] and board[0][0] != ' ':
        return True, board[0][0]

    if board[0][2] == board[1][1] == board[2][0] and board[0][2] != ' ':
        return True, board[0][2]

    return False, None

def play(bboxes):
    grid_state = infer_tic_tac_toe_state(bboxes)
    move, player_letter = find_best_move(grid_state)
    if not move:
        print("fini game over")
    else:
        position = get_cell_center(move, bboxes['grid'][0])
        return position



@app.route('/play', methods=['POST'])
def play():
    if request.method == 'POST':
        try:
            data = request.json
            position, letter = get_next_move(data['bounding_boxes'])
            response = {
                "position": position,
                "letter": letter
            }
            print(response, letter)
            return jsonify(response), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500
        


if __name__ == "__main__":
    app.run(debug=True)