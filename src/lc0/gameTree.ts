import { boardToFen, parseFen, START_FEN, type BoardState } from '../chess/board.ts';
import { legalMoves, makeMove } from '../chess/movegen.ts';
import { moveToUci, type Move } from '../chess/moveCodec.ts';
import { moveToSan } from '../chess/san.ts';

/**
 * A branching game/variation tree, the shared spine for the analysis board and
 * single-engine play modes. Each node stores the move that led to it (from its
 * parent) and the resulting FEN; children[0] is the mainline continuation and
 * additional children are variations.
 */
export interface GameNode {
  id: number;
  parent: GameNode | null;
  /** Move from the parent position that produced this node; null at the root. */
  move: Move | null;
  /** SAN of `move` in the parent position; null at the root. */
  san: string | null;
  fen: string;
  /** Half-move distance from the root. */
  ply: number;
  children: GameNode[];
  comment?: string;
}

export class GameTree {
  readonly root: GameNode;
  current: GameNode;
  private nextId = 1;

  constructor(startFen: string = START_FEN) {
    this.root = { id: 0, parent: null, move: null, san: null, fen: boardToFen(parseFen(startFen)), ply: 0, children: [] };
    this.current = this.root;
  }

  currentBoard(): BoardState {
    return parseFen(this.current.fen);
  }

  /** Add a move from the current node, reusing an existing child if present. Advances current. */
  addMove(move: Move): GameNode {
    const uci = moveToUci(move);
    const existing = this.current.children.find((child) => child.move && moveToUci(child.move) === uci);
    if (existing) {
      this.current = existing;
      return existing;
    }
    const board = parseFen(this.current.fen);
    const san = moveToSan(board, move);
    const childBoard = makeMove(board, move);
    const node: GameNode = {
      id: this.nextId++,
      parent: this.current,
      move,
      san,
      fen: boardToFen(childBoard),
      ply: this.current.ply + 1,
      children: [],
    };
    this.current.children.push(node);
    this.current = node;
    return node;
  }

  /** Add a move given as UCI; returns null if it is not legal in the current position. */
  addUci(uci: string): GameNode | null {
    const move = legalMoves(parseFen(this.current.fen)).find((candidate) => moveToUci(candidate) === uci);
    return move ? this.addMove(move) : null;
  }

  goTo(node: GameNode): void {
    this.current = node;
  }

  toStart(): void {
    this.current = this.root;
  }

  toEnd(): void {
    let node = this.current;
    while (node.children[0]) node = node.children[0];
    this.current = node;
  }

  back(): boolean {
    if (!this.current.parent) return false;
    this.current = this.current.parent;
    return true;
  }

  forward(childIndex = 0): boolean {
    const next = this.current.children[childIndex];
    if (!next) return false;
    this.current = next;
    return true;
  }

  /** Mainline nodes (root excluded) following children[0] from `from`. */
  mainlineFrom(from: GameNode = this.root): GameNode[] {
    const line: GameNode[] = [];
    let node: GameNode | undefined = from.children[0];
    while (node) {
      line.push(node);
      node = node.children[0];
    }
    return line;
  }

  /** Path of nodes from the root to `node` inclusive (root first). */
  lineTo(node: GameNode = this.current): GameNode[] {
    const path: GameNode[] = [];
    let cursor: GameNode | null = node;
    while (cursor) {
      path.unshift(cursor);
      cursor = cursor.parent;
    }
    return path;
  }

  /** Board states from the root to `node` inclusive — the LC0 evaluator history input. */
  historyBoards(node: GameNode = this.current): BoardState[] {
    return this.lineTo(node).map((entry) => parseFen(entry.fen));
  }

  /** Make `node` its parent's mainline continuation (move it to children[0]). */
  promote(node: GameNode): void {
    const parent = node.parent;
    if (!parent) return;
    const index = parent.children.indexOf(node);
    if (index <= 0) return;
    parent.children.splice(index, 1);
    parent.children.unshift(node);
  }

  /** Remove `node` and its subtree. If current was inside it, fall back to the parent. */
  remove(node: GameNode): void {
    const parent = node.parent;
    if (!parent) return;
    const index = parent.children.indexOf(node);
    if (index < 0) return;
    parent.children.splice(index, 1);
    if (this.containsNode(node, this.current)) this.current = parent;
  }

  private containsNode(ancestor: GameNode, candidate: GameNode): boolean {
    let cursor: GameNode | null = candidate;
    while (cursor) {
      if (cursor === ancestor) return true;
      cursor = cursor.parent;
    }
    return false;
  }
}
