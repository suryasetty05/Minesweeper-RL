import numpy as np
import itertools
from collections import defaultdict

def neighbors(board, i, j):
    height, width = board.shape

    for x in range(max(0, i - 1), min(height, i + 2)):
        for y in range(max(0, j - 1), min(width, j + 2)):
            if (x, y) != (i, j):
                yield (x, y)

class MinesweeperLogic:
    def __init__(self, board, mines_left=None):
        self.board = board
        self.height, self.width = board.shape
        self.total_mines = mines_left
        
        # Internal state
        self.safe = set()
        self.mines = set()
        self.probs = {}
        
    def solve(self, enum_limit = 22):
        # Iterative Set Difference Propagation
        self._run_set_difference_engine()
        
        # Build remaining coupled components
        components, constraints = self._build_coupled_components()
        
        self.probs = {}
        all_unknowns = []

        for r in range(self.height):
            for c in range(self.width):
                if self.board[r, c] == -1 and (r, c) not in self.safe and (r, c) not in self.mines:
                    all_unknowns.append((r, c))
        
        solved_unknowns = set()
        
        for comp in components:
            comp_size = len(comp)
            if comp_size <= enum_limit:
                # Enumeration limit to increase computational efficiency
                comp_probs = self._enumerate_component(comp, constraints)
                self.probs.update(comp_probs)
            else:
                # Heuristic fallback; probability estimate with all unsolved cells
                mines_remaining = self.total_mines - len(self.mines) if self.total_mines else 0
                unknowns_remaining = len(all_unknowns)

                p_fallback = mines_remaining / unknowns_remaining if unknowns_remaining > 0 else 0.5

                # Debugging
                print(f'WARNING: Component size {comp_size} exceeds limit ({enum_limit}). Using fallback P={p_fallback:.4f}.') 
                for cell in comp:
                    self.probs[cell] = p_fallback
            
            solved_unknowns.update(comp)
        
        # Probabilities after process
        for cell, p in self.probs.items():
            if p <= 1e-6:
                self.safe.add(cell)
            elif p >= 1.0 - 1e-6:
                self.mines.add(cell)

        # Handle cells not touching any number
        background_cells = [x for x in all_unknowns if x not in solved_unknowns]
        if background_cells:
            mines_remaining = self.total_mines - len(self.mines) if self.total_mines else 0
            unknowns_remaining = len(background_cells) + sum(1 for c, p in self.probs.items() if 1e-6 < p < 1.0 - 1e-6)
            
            p_bg = 0.5 
            if self.total_mines and unknowns_remaining > 0:
                expected_in_front = sum(self.probs.values())
                mines_for_bg_area = self.total_mines - len(self.mines) - expected_in_front
                mines_for_bg_area = max(0, mines_for_bg_area)
                
                if len(background_cells) > 0:
                     p_bg = mines_for_bg_area / len(background_cells)
            
            for c in background_cells:
                self.probs[c] = p_bg

        return list(self.safe), list(self.mines), self.probs

    def _run_set_difference_engine(self):
        changed = True
        while changed:
            changed = False

            # Currently active constraints
            active_constraints = []
            updates = set() 
            
            for r in range(self.height):
                for c in range(self.width):
                    val = self.board[r, c]
                    if val >= 0:
                        nk = list(neighbors(self.board, r, c))
                        unknowns = set()
                        curr_mines = 0
                        
                        for n in nk:
                            if n in self.mines:
                                curr_mines += 1
                            elif n in self.safe:
                                pass # Safe
                            elif self.board[n[0], n[1]] == -1:
                                unknowns.add(n)
                        
                        rem_val = val - curr_mines
                        
                        if len(unknowns) > 0:
                            if rem_val == 0:
                                # All unknowns are safe
                                for u in unknowns:
                                    if u not in self.safe:
                                        self.safe.add(u)
                                        changed = True
                            elif rem_val == len(unknowns):
                                # All unknowns are mines
                                for u in unknowns:
                                    if u not in self.mines:
                                        self.mines.add(u)
                                        changed = True
                            else:
                                # Still ambiguous, add to constraints
                                active_constraints.append((unknowns, rem_val))
            
            if changed: continue # Restart loop, propagate simple stuff first

            # Deduplicate constraints
            unique_cons = []
            seen_cons = set()

            for u, r in active_constraints:
                fs = frozenset(u)
                if (fs, r) not in seen_cons:
                    unique_cons.append((u, r))
                    seen_cons.add((fs, r))
            
            active_constraints = unique_cons
            N = len(active_constraints)
            
            for i in range(N):
                set_a, val_a = active_constraints[i]
                
                for j in range(i + 1, N):
                    set_b, val_b = active_constraints[j]
                    
                    # A subset B
                    if set_a.issubset(set_b):
                        diff = set_b - set_a
                        diff_val = val_b - val_a
                        if len(diff) > 0:
                            if diff_val == 0:
                                for x in diff:
                                    if x not in self.safe:
                                        self.safe.add(x)
                                        changed = True
                            elif diff_val == len(diff):
                                for x in diff:
                                    if x not in self.mines:
                                        self.mines.add(x)
                                        changed = True
                    
                    # B subset A
                    elif set_b.issubset(set_a):
                        diff = set_a - set_b
                        diff_val = val_a - val_b
                        if len(diff) > 0:
                            if diff_val == 0:
                                for x in diff:
                                    if x not in self.safe:
                                        self.safe.add(x)
                                        changed = True
                            elif diff_val == len(diff):
                                for x in diff:
                                    if x not in self.mines:
                                        self.mines.add(x)
                                        changed = True
            
            if changed: continue

    def _build_coupled_components(self):
        # Constraint: ({set of cells}, count)
        cons_list = []
        cell_to_cons = defaultdict(list)
        all_unknowns = set()
        
        for r in range(self.height):
            for c in range(self.width):
                val = self.board[r, c]
                if val >= 0:
                    nk = list(neighbors(self.board, r, c))
                    u_neighbors = []
                    found_mines = 0
                    for n in nk:
                        if n in self.mines: 
                            found_mines += 1
                        elif n in self.safe: 
                            pass
                        elif self.board[n[0], n[1]] == -1: 
                            u_neighbors.append(n)
                    
                    rem = val - found_mines
                    if u_neighbors:
                        cid = len(cons_list)
                        cons_entry = (set(u_neighbors), rem)
                        cons_list.append(cons_entry)

                        for u in u_neighbors:
                            all_unknowns.add(u)
                            cell_to_cons[u].append(cid)

        # BFS to find components
        seen = set()
        components = []
        for cell in all_unknowns:
            if cell in seen: 
                continue
            
            q = [cell]
            seen.add(cell)
            comp = {cell}
            
            idx = 0
            while idx < len(q):
                curr = q[idx]; idx+=1

                for cid in cell_to_cons[curr]:
                    # Get all cells in that constraint
                    c_cells, _ = cons_list[cid]
                    for neighbor in c_cells:
                        if neighbor not in seen:
                            seen.add(neighbor)
                            comp.add(neighbor)
                            q.append(neighbor)

            components.append(list(comp))
            
        return components, cons_list

    def _enumerate_component(self, comp, all_constraints):
        # Brute force exact probabilities for component
        comp_set = set(comp)
        relevant_cons = []

        for cells, count in all_constraints:
            if not cells.isdisjoint(comp_set):
                relevant_cons.append((list(cells), count))
        
        # Map cells for optimization
        c_to_idx = {c: i for i, c in enumerate(comp)}
        N = len(comp)

        fast_cons = []
        for cells, count in relevant_cons:
            ind = [c_to_idx[c] for c in cells if c in c_to_idx]
            if ind:
                fast_cons.append((ind, count))
        
        # Valid solutions N
        solutions = 0
        counts = [0] * N
        
        # Recursive solver
        def solve_rec(idx, current_assign):
            nonlocal solutions
            
            # Pruning check
            for c_indices, c_target in fast_cons:
                curr_sum = 0
                unk_count = 0
                for ci in c_indices:
                    if ci < idx:
                        curr_sum += current_assign[ci]
                    else:
                        unk_count += 1

                if curr_sum > c_target: 
                    return

                if curr_sum + unk_count < c_target: 
                    return

            if idx == N:
                solutions += 1
                for i in range(N):
                    if current_assign[i]:
                        counts[i] += 1
                return

            # Try 0
            current_assign[idx] = 0
            solve_rec(idx + 1, current_assign)
            
            # Try 1
            current_assign[idx] = 1
            solve_rec(idx + 1, current_assign)
            
        # Run recursion
        solve_rec(0, [0] * N)
        
        probs = {}
        if solutions == 0:
            for c in comp: probs[c] = 0.5
        else:
            for i in range(N):
                probs[comp[i]] = counts[i] / solutions
                
        return probs

def expert_solve(board, mines_left = None, enum_limit = 22):
    solver = MinesweeperLogic(board, mines_left)
    s, m, p = solver.solve(enum_limit = enum_limit)

    return solver.solve()

def deterministic_solver(board):
    s, m, p = expert_solve(board)
    
    # Deterministic
    if s:
        return s[0]
    
    # Probabilistic
    candidates = []
    for cell, prob in p.items():
        if prob is not None and prob < 0.99: # Small float margin
            candidates.append((prob, cell))
            
    if candidates:
        candidates.sort(key = lambda x: x[0])
        best_prob = candidates[0][0]
        
        best_moves = [c for prob, c in candidates if abs(prob - best_prob) < 1e-5]
        idx = np.random.choice(len(best_moves))

        return best_moves[idx]
    
    # Opening move: (1, 1)
    unknown_indices = np.argwhere(board == -1)
    if len(unknown_indices) > 0:
        return (1, 1)
    
    return None