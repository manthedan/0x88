use crate::board::{
    file, idx, on, rank, Board, Color, Move, Piece, Role, BISHOP, KING, KNIGHT, ROOK,
};

fn push_step(board: &Board, moves: &mut Vec<Move>, from: u8, df: i8, dr: i8) {
    let (f, r) = (file(from) + df, rank(from) + dr);
    if !on(f, r) {
        return;
    }
    let to = idx(f, r);
    if board.squares[to as usize].map(|p| p.color) != Some(board.turn) {
        moves.push(Move {
            from,
            to,
            promotion: None,
        });
    }
}

fn push_slides(board: &Board, moves: &mut Vec<Move>, from: u8, dirs: &[(i8, i8)]) {
    for &(df, dr) in dirs {
        let (mut f, mut r) = (file(from) + df, rank(from) + dr);
        while on(f, r) {
            let to = idx(f, r);
            match board.squares[to as usize] {
                None => moves.push(Move {
                    from,
                    to,
                    promotion: None,
                }),
                Some(piece) => {
                    if piece.color != board.turn {
                        moves.push(Move {
                            from,
                            to,
                            promotion: None,
                        });
                    }
                    break;
                }
            }
            f += df;
            r += dr;
        }
    }
}

fn can_castle(board: &Board, mask: u8) -> bool {
    if board.castling & mask == 0 {
        return false;
    }
    let white = mask == 1 || mask == 2;
    let color = if white { Color::White } else { Color::Black };
    let r = if white { 0 } else { 7 };
    let king_from = idx(4, r);
    let rook_from = if mask == 1 || mask == 4 {
        idx(7, r)
    } else {
        idx(0, r)
    };
    if board.turn != color
        || board.squares[king_from as usize]
            != Some(Piece {
                color,
                role: Role::King,
            })
        || board.squares[rook_from as usize]
            != Some(Piece {
                color,
                role: Role::Rook,
            })
    {
        return false;
    }
    let between: &[u8] = if mask == 1 || mask == 4 {
        &[idx(5, r), idx(6, r)]
    } else {
        &[idx(1, r), idx(2, r), idx(3, r)]
    };
    if between.iter().any(|&s| board.squares[s as usize].is_some()) {
        return false;
    }
    let pass: &[u8] = if mask == 1 || mask == 4 {
        &[idx(4, r), idx(5, r), idx(6, r)]
    } else {
        &[idx(4, r), idx(3, r), idx(2, r)]
    };
    pass.iter()
        .all(|&s| !is_square_attacked(board, s, color.opposite()))
}

pub fn pseudo_legal_moves(board: &Board) -> Vec<Move> {
    let mut moves = Vec::with_capacity(64);
    for from in 0u8..64 {
        let Some(piece) = board.squares[from as usize] else {
            continue;
        };
        if piece.color != board.turn {
            continue;
        }
        match piece.role {
            Role::Pawn => {
                let dir = if board.turn == Color::White { 1 } else { -1 };
                let start = if board.turn == Color::White { 1 } else { 6 };
                let promo = if board.turn == Color::White { 7 } else { 0 };
                let (f, r) = (file(from), rank(from));
                let one = r + dir;
                if on(f, one) && board.squares[idx(f, one) as usize].is_none() {
                    let to = idx(f, one);
                    if one == promo {
                        for promotion in [Role::Queen, Role::Rook, Role::Bishop, Role::Knight] {
                            moves.push(Move {
                                from,
                                to,
                                promotion: Some(promotion),
                            });
                        }
                    } else {
                        moves.push(Move {
                            from,
                            to,
                            promotion: None,
                        });
                    }
                    let two = r + 2 * dir;
                    if r == start && on(f, two) && board.squares[idx(f, two) as usize].is_none() {
                        moves.push(Move {
                            from,
                            to: idx(f, two),
                            promotion: None,
                        });
                    }
                }
                for df in [-1, 1] {
                    let cf = f + df;
                    if !on(cf, one) {
                        continue;
                    }
                    let to = idx(cf, one);
                    let capture = board.squares[to as usize].map(|p| p.color)
                        == Some(board.turn.opposite())
                        || Some(to) == board.ep_square;
                    if capture {
                        if one == promo {
                            for promotion in [Role::Queen, Role::Rook, Role::Bishop, Role::Knight] {
                                moves.push(Move {
                                    from,
                                    to,
                                    promotion: Some(promotion),
                                });
                            }
                        } else {
                            moves.push(Move {
                                from,
                                to,
                                promotion: None,
                            });
                        }
                    }
                }
            }
            Role::Knight => {
                for &(df, dr) in KNIGHT {
                    push_step(board, &mut moves, from, df, dr);
                }
            }
            Role::Bishop => push_slides(board, &mut moves, from, BISHOP),
            Role::Rook => push_slides(board, &mut moves, from, ROOK),
            Role::Queen => {
                push_slides(board, &mut moves, from, BISHOP);
                push_slides(board, &mut moves, from, ROOK);
            }
            Role::King => {
                for &(df, dr) in KING {
                    push_step(board, &mut moves, from, df, dr);
                }
                if can_castle(board, if board.turn == Color::White { 1 } else { 4 }) {
                    moves.push(Move {
                        from,
                        to: idx(6, if board.turn == Color::White { 0 } else { 7 }),
                        promotion: None,
                    });
                }
                if can_castle(board, if board.turn == Color::White { 2 } else { 8 }) {
                    moves.push(Move {
                        from,
                        to: idx(2, if board.turn == Color::White { 0 } else { 7 }),
                        promotion: None,
                    });
                }
            }
        }
    }
    moves
}

pub fn is_square_attacked(board: &Board, sq: u8, by: Color) -> bool {
    let (f, r) = (file(sq), rank(sq));
    let pawn_rank = r + if by == Color::White { -1 } else { 1 };
    for df in [-1, 1] {
        let pf = f + df;
        if on(pf, pawn_rank)
            && board.squares[idx(pf, pawn_rank) as usize]
                == Some(Piece {
                    color: by,
                    role: Role::Pawn,
                })
        {
            return true;
        }
    }
    for &(df, dr) in KNIGHT {
        let (nf, nr) = (f + df, r + dr);
        if on(nf, nr)
            && board.squares[idx(nf, nr) as usize]
                == Some(Piece {
                    color: by,
                    role: Role::Knight,
                })
        {
            return true;
        }
    }
    for &(df, dr) in BISHOP {
        if ray_attacked(board, f, r, df, dr, by, Role::Bishop) {
            return true;
        }
    }
    for &(df, dr) in ROOK {
        if ray_attacked(board, f, r, df, dr, by, Role::Rook) {
            return true;
        }
    }
    for &(df, dr) in KING {
        let (kf, kr) = (f + df, r + dr);
        if on(kf, kr)
            && board.squares[idx(kf, kr) as usize]
                == Some(Piece {
                    color: by,
                    role: Role::King,
                })
        {
            return true;
        }
    }
    false
}

fn ray_attacked(board: &Board, f: i8, r: i8, df: i8, dr: i8, by: Color, slider: Role) -> bool {
    let (mut sf, mut sr) = (f + df, r + dr);
    while on(sf, sr) {
        if let Some(p) = board.squares[idx(sf, sr) as usize] {
            return p.color == by && (p.role == slider || p.role == Role::Queen);
        }
        sf += df;
        sr += dr;
    }
    false
}

pub fn king_square(board: &Board, color: Color) -> Option<u8> {
    board
        .squares
        .iter()
        .position(|&p| {
            p == Some(Piece {
                color,
                role: Role::King,
            })
        })
        .map(|i| i as u8)
}

pub fn in_check(board: &Board, color: Color) -> bool {
    king_square(board, color)
        .map(|sq| is_square_attacked(board, sq, color.opposite()))
        .unwrap_or(true)
}

pub fn legal_moves(board: &Board) -> Vec<Move> {
    pseudo_legal_moves(board)
        .into_iter()
        .filter(|&m| !in_check(&make_move(board, m), board.turn))
        .collect()
}

fn remove_castling(c: &mut u8, mask: u8) {
    *c &= !mask;
}

pub fn make_move(board: &Board, m: Move) -> Board {
    let mut next = board.clone();
    let piece = next.squares[m.from as usize].expect("No piece on source square");
    let captured = next.squares[m.to as usize];
    let is_pawn = piece.role == Role::Pawn;
    let is_castle = piece.role == Role::King && (file(m.to) - file(m.from)).abs() == 2;
    let is_ep = is_pawn
        && Some(m.to) == board.ep_square
        && captured.is_none()
        && file(m.from) != file(m.to);
    next.squares[m.from as usize] = None;
    if is_ep {
        next.squares[idx(file(m.to), rank(m.from)) as usize] = None;
    }
    next.squares[m.to as usize] = Some(Piece {
        color: board.turn,
        role: m.promotion.unwrap_or(piece.role),
    });
    if is_castle {
        let r = if board.turn == Color::White { 0 } else { 7 };
        if file(m.to) == 6 {
            next.squares[idx(5, r) as usize] = next.squares[idx(7, r) as usize];
            next.squares[idx(7, r) as usize] = None;
        } else {
            next.squares[idx(3, r) as usize] = next.squares[idx(0, r) as usize];
            next.squares[idx(0, r) as usize] = None;
        }
    }
    match piece {
        Piece {
            color: Color::White,
            role: Role::King,
        } => remove_castling(&mut next.castling, 1 | 2),
        Piece {
            color: Color::Black,
            role: Role::King,
        } => remove_castling(&mut next.castling, 4 | 8),
        _ => {}
    }
    for (sq, mask) in [(0, 2), (7, 1), (56, 8), (63, 4)] {
        if m.from == sq || m.to == sq {
            remove_castling(&mut next.castling, mask);
        }
    }
    next.turn = board.turn.opposite();
    next.ep_square = if is_pawn && (rank(m.to) - rank(m.from)).abs() == 2 {
        Some(idx(file(m.from), (rank(m.from) + rank(m.to)) / 2))
    } else {
        None
    };
    next.halfmove = if is_pawn || captured.is_some() || is_ep {
        0
    } else {
        board.halfmove + 1
    };
    next.fullmove = board.fullmove + if board.turn == Color::Black { 1 } else { 0 };
    next
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{move_to_uci, parse_fen, Piece, START_FEN};

    #[test]
    fn start_position_has_20_legal_moves() {
        assert_eq!(legal_moves(&parse_fen(START_FEN).unwrap()).len(), 20);
    }

    #[test]
    fn rejects_king_into_check() {
        let b = parse_fen("k3r3/8/8/8/8/8/8/4K3 w - - 0 1").unwrap();
        let moves: Vec<String> = legal_moves(&b).into_iter().map(move_to_uci).collect();
        assert!(!moves.contains(&"e1e2".to_string()));
    }

    #[test]
    fn castling_and_ep() {
        let b = parse_fen("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1").unwrap();
        let moves: Vec<String> = legal_moves(&b).into_iter().map(move_to_uci).collect();
        assert!(moves.contains(&"e1g1".to_string()));
        assert!(moves.contains(&"e1c1".to_string()));
        let ep = make_move(
            &parse_fen("8/8/8/3pP3/8/8/8/4K2k w - d6 0 1").unwrap(),
            Move {
                from: 36,
                to: 43,
                promotion: None,
            },
        );
        assert!(ep.squares[35].is_none());
        assert_eq!(
            ep.squares[43],
            Some(Piece {
                color: Color::White,
                role: Role::Pawn
            })
        );
    }
}
