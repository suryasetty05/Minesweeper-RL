"""
A rule-based logic solver for Minesweeper-Env written by Gemini 3
"""
import numpy as np
import itertools
from collections import defaultdict

def neighbors(board, i, j):
    H, W = board.shape
    for x in range(max(0, i-1), min(H, i+2)):
        for y in range(max(0, j-1), min(W, j+2)):
            if (x,y) != (i,j):
                yield (x,y)

class MinesweeperLogic:
    def __init__(self, board, mines_left=None):
        self.board = board
        self.H, self.W = board.shape
        self.total_mines = mines_left # Optional, for global probability
        
        # Internal state
        self.safe = set()
        self.mines = set()
        self.probs = {} # (x,y) -> float
        
    def solve(self, enum_limit=22):
        """
        Main execution pipeline.
        Returns: (list_of_safe, list_of_mines, prob_dict)
        """
        # 1. Basic & Set Difference Propagation (Iterative)
        self._run_set_difference_engine()
        
        # 2. Build remaining coupled components
        components, constraints = self._build_coupled_components() # 'components' is defined here
        
        # 3. Calculate probabilities
        self.probs = {}
        
        # Track global unknown count for background probability
        all_unknowns = []
        for r in range(self.H):
            for c in range(self.W):
                if self.board[r,c] == -1 and (r,c) not in self.safe and (r,c) not in self.mines:
                    all_unknowns.append((r,c))
        
        solved_unknowns = set()
        
        # --- PHASE 3 MODIFICATION START ---
        for comp in components:
            
            comp_size = len(comp)
            
            if comp_size <= enum_limit:
                # OPTION A: EXACT ENUMERATION (Fast and accurate for small components)
                # We perform exact enumeration on components because Set Difference
                # usually breaks the board into very small independent chunks.
                comp_probs = self._enumerate_component(comp, constraints)
                self.probs.update(comp_probs)
            else:
                # OPTION B: HEURISTIC FALLBACK (Necessary for scalability)
                # If the component is too large, we use a probability estimate 
                # (density) to avoid $O(2^N)$ runtime.
                
                # Estimate remaining mines globally
                mines_remaining = self.total_mines - len(self.mines) if self.total_mines else 0
                unknowns_remaining = len(all_unknowns) # includes all unsolved cells
                
                # Calculate the global density (a safe heuristic for large components)
                p_fallback = mines_remaining / unknowns_remaining if unknowns_remaining > 0 else 0.5

                print(f"WARNING: Component size {comp_size} exceeds limit ({enum_limit}). Using fallback P={p_fallback:.4f}.") # Important for debugging
                
                for cell in comp:
                    self.probs[cell] = p_fallback
            
            solved_unknowns.update(comp)

        # --- PHASE 3 MODIFICATION END ---
        
        # 4. Post-process probabilities
        for cell, p in self.probs.items():
            if p <= 1e-6:
                self.safe.add(cell)
            elif p >= 1.0 - 1e-6:
                self.mines.add(cell)

        # 5. Handle "Background" cells (cells not touching any number)
        background_cells = [x for x in all_unknowns if x not in solved_unknowns]
        
        if background_cells:
            # Estimate remaining mines (re-calculated to be safer)
            mines_remaining = self.total_mines - len(self.mines) if self.total_mines else 0
            unknowns_remaining = len(background_cells) + sum(1 for c,p in self.probs.items() if 1e-6 < p < 1.0 - 1e-6)
            
            p_bg = 0.5 
            if self.total_mines and unknowns_remaining > 0:
                # Subtract expected mines found in the front line
                expected_in_front = sum(self.probs.values())
                mines_for_bg_area = self.total_mines - len(self.mines) - expected_in_front
                mines_for_bg_area = max(0, mines_for_bg_area)
                
                if len(background_cells) > 0:
                     p_bg = mines_for_bg_area / len(background_cells)
            
            for c in background_cells:
                self.probs[c] = p_bg

        return list(self.safe), list(self.mines), self.probs

    def _run_set_difference_engine(self):
        """
        Iteratively applies single-cell constraints AND set-subset constraints.
        Example: If A = {x,y} has 1 mine, and B = {x,y,z} has 1 mine,
        then B-A = {z} has 0 mines.
        """
        changed = True
        while changed:
            changed = False
            
            # Step A: Collect current active constraints
            # map: (r,c) of number -> (set of unknown neighbors, remaining mines)
            active_constraints = []
            
            # Also keep track of direct deductions to apply immediately
            updates = set() 
            
            for r in range(self.H):
                for c in range(self.W):
                    val = self.board[r,c]
                    if val >= 0:
                        nk = list(neighbors(self.board, r, c))
                        unknowns = set()
                        curr_mines = 0
                        
                        for n in nk:
                            if n in self.mines:
                                curr_mines += 1
                            elif n in self.safe:
                                pass # is safe, ignore
                            elif self.board[n[0], n[1]] == -1: # raw unknown
                                unknowns.add(n)
                            # else: revealed number, ignore
                        
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
                                # Still ambiguous, add to constraints for set diff
                                active_constraints.append((unknowns, rem_val))
            
            if changed: continue # Restart loop to propagate simple stuff first

            # Step B: Advanced Set Difference
            # Compare every pair of constraints
            # Optimization: Only compare if they share at least one cell (heuristic)
            # or just N^2 for small N (number of edge constraints is usually < 100)
            
            # Deduplicate constraints
            unique_cons = []
            seen_cons = set()
            for u, r in active_constraints:
                fz = frozenset(u)
                if (fz, r) not in seen_cons:
                    unique_cons.append((u, r))
                    seen_cons.add((fz, r))
            
            active_constraints = unique_cons
            N = len(active_constraints)
            
            for i in range(N):
                set_a, val_a = active_constraints[i]
                for j in range(i + 1, N):
                    set_b, val_b = active_constraints[j]
                    
                    # Check A subset B
                    if set_a.issubset(set_b):
                        diff = set_b - set_a
                        diff_val = val_b - val_a
                        if len(diff) > 0:
                            if diff_val == 0:
                                for x in diff:
                                    if x not in self.safe:
                                        self.safe.add(x); changed = True
                            elif diff_val == len(diff):
                                for x in diff:
                                    if x not in self.mines:
                                        self.mines.add(x); changed = True
                    
                    # Check B subset A
                    elif set_b.issubset(set_a):
                        diff = set_a - set_b
                        diff_val = val_a - val_b
                        if len(diff) > 0:
                            if diff_val == 0:
                                for x in diff:
                                    if x not in self.safe:
                                        self.safe.add(x); changed = True
                            elif diff_val == len(diff):
                                for x in diff:
                                    if x not in self.mines:
                                        self.mines.add(x); changed = True
            
            if changed: continue

    def _build_coupled_components(self):
        """
        Builds graph of unknown cells connected by shared constraints.
        Returns list of sets (components), and map of relevant constraints.
        """
        # Re-scan board for constraints
        # Constraint: ({set of cells}, count)
        cons_list = []
        cell_to_cons = defaultdict(list)
        all_unknowns = set()
        
        for r in range(self.H):
            for c in range(self.W):
                val = self.board[r,c]
                if val >= 0:
                    nk = list(neighbors(self.board, r, c))
                    u_neigh = []
                    found_mines = 0
                    for n in nk:
                        if n in self.mines: found_mines += 1
                        elif n in self.safe: pass
                        elif self.board[n[0], n[1]] == -1: u_neigh.append(n)
                    
                    rem = val - found_mines
                    if u_neigh:
                        cid = len(cons_list)
                        cons_entry = (set(u_neigh), rem)
                        cons_list.append(cons_entry)
                        for u in u_neigh:
                            all_unknowns.add(u)
                            cell_to_cons[u].append(cid)

        # BFS to find components
        seen = set()
        components = []
        for cell in all_unknowns:
            if cell in seen: continue
            
            q = [cell]
            seen.add(cell)
            comp = {cell}
            
            idx = 0
            while idx < len(q):
                curr = q[idx]; idx+=1
                # get all constraints touching this cell
                for cid in cell_to_cons[curr]:
                    # get all cells in that constraint
                    c_cells, _ = cons_list[cid]
                    for neighbor in c_cells:
                        if neighbor not in seen:
                            seen.add(neighbor)
                            comp.add(neighbor)
                            q.append(neighbor)
            components.append(list(comp))
            
        return components, cons_list

    def _enumerate_component(self, comp, all_constraints):
        """
        Brute force exact probabilities for a component.
        """
        # Identify relevant constraints
        comp_set = set(comp)
        relevant_cons = []
        for cells, count in all_constraints:
            # If constraint overlaps with component
            if not cells.isdisjoint(comp_set):
                # We only care about the variables inside this component.
                # Usually, constraints won't span multiple components 
                # because components are defined by overlap.
                
                # Filter constraint to only this component's cells?
                # Actually, due to logic, cells in constraint MUST be in component 
                # (or they would have bridged the components).
                relevant_cons.append((list(cells), count))
        
        # Optimize: Map cells to 0..N-1
        c_to_idx = {c: i for i, c in enumerate(comp)}
        N = len(comp)
        
        # Pre-process constraints into index lists
        # (indices, target_sum)
        fast_cons = []
        for cells, count in relevant_cons:
            ind = [c_to_idx[c] for c in cells if c in c_to_idx]
            if ind:
                fast_cons.append((ind, count))
        
        # Valid solutions count
        solutions = 0
        counts = [0] * N
        
        # Backtracking / Recursion is faster than itertools.product for pruning
        # We define a recursive solver
        
        def solve_rec(idx, current_assign):
            nonlocal solutions
            
            # Pruning check
            # For every constraint, check if it's already violated or satisfied
            for c_indices, c_target in fast_cons:
                curr_sum = 0
                unk_count = 0
                for ci in c_indices:
                    if ci < idx:
                        curr_sum += current_assign[ci]
                    else:
                        unk_count += 1
                
                # If we have exceeded target
                if curr_sum > c_target: return
                # If we can't possibly reach target
                if curr_sum + unk_count < c_target: return

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
        solve_rec(0, [0]*N)
        
        probs = {}
        if solutions == 0:
            # Inconsistent state (likely due to flag error in input or impossible board)
            # Return uniform 0.5 (or handle error)
            for c in comp: probs[c] = 0.5
        else:
            for i in range(N):
                probs[comp[i]] = counts[i] / solutions
                
        return probs

def expert_solve(board, mines_left=None, enum_limit=22):
    """
    Wrapper to match your API.
    """
    solver = MinesweeperLogic(board, mines_left)
    s, m, p = solver.solve(enum_limit=enum_limit) # Pass the limit here
    return solver.solve()

def deterministic_solver(board):
    """
    Combined Safe > Min-Risk-Prob > Random
    """
    s, m, p = expert_solve(board)
    
    # 1. Deterministic Safe
    if s:
        return s[0]
    
    # 2. Probabilistic: Lowest mine probability
    # Filter out known mines (p=1.0) and None
    candidates = []
    for cell, prob in p.items():
        if prob is not None and prob < 0.99: # Allow small float margin
            candidates.append((prob, cell))
            
    if candidates:
        # Sort by probability ASC
        candidates.sort(key=lambda x: x[0])
        best_prob = candidates[0][0]
        # Pick randomly among the ties for best probability to avoid bias
        best_moves = [c for prob, c in candidates if abs(prob - best_prob) < 1e-5]
        idx = np.random.choice(len(best_moves))
        return best_moves[idx]
    
    # 3. Fallback: Pure Random (Opening move)
    unknown_indices = np.argwhere(board == -1)
    if len(unknown_indices) > 0:
        idx = np.random.randint(len(unknown_indices))
        return tuple(unknown_indices[idx])
    
    return None

if __name__ == "__main__":
    # Corrected Board:
    # We use '1' on the edges, because if the corners are mines 
    # (as expected in 1-2-1), the edge neighbors would see 1 mine.
    # Alternatively, use unrevealed (-1) or boundaries.
    
    # [1, 1, 2, 1, 1]
    # [?, ?, ?, ?, ?]
    
    b = np.array([
        [0, 0, 0, 0, 0],
        [1, 1, 2, 1, 1], 
        [0, -1, -1, -1, 0],
        [0, -1, -1, -1, 0]
    ])
    
    # Update: We must ensure the 0s in row 0 don't interfere with row 2.
    # Row 0 neighbors Row 1. It does NOT neighbor Row 2.
    # So the 0s in Row 0 are fine. 
    # However, the 0s in Row 2 and 3 (columns 0 and 4) interact with Row 1.
    
    # Let's make a cleaner isolated board using -2 (Wall) to avoid any edge noise.
    # -2 is not processed by the solver.
    b_clean = np.array([
        [1,  2,  1],
        [-1, -1, -1],
    ])

    print("Solving 1-2-1 Pattern (Clean Board)...")
    s, m, p = expert_solve(b_clean)
    print("Safe:", s) 
    print("Mines:", m)
    # Filter out the wall (-2) probabilities
    print("Probs:", {k:round(v, 2) for k,v in p.items() if v is not None})