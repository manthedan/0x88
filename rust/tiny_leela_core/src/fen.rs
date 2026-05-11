use crate::board::{idx, square_index, square_name, Board, Color, Piece, Role};

fn piece_from_fen(ch: char) -> Option<Piece> {
    let color = if ch.is_ascii_uppercase() {
        Color::White
    } else {
        Color::Black
    };
    let role = match ch.to_ascii_lowercase() {
        'p' => Role::Pawn,
        'n' => Role::Knight,
        'b' => Role::Bishop,
        'r' => Role::Rook,
        'q' => Role::Queen,
        'k' => Role::King,
        _ => return None,
    };
    Some(Piece { color, role })
}

fn piece_to_fen(piece: Piece) -> char {
    let ch = match piece.role {
        Role::Pawn => 'p',
        Role::Knight => 'n',
        Role::Bishop => 'b',
        Role::Rook => 'r',
        Role::Queen => 'q',
        Role::King => 'k',
    };
    if piece.color == Color::White {
        ch.to_ascii_uppercase()
    } else {
        ch
    }
}

pub fn parse_fen(fen: &str) -> Result<Board, String> {
    let mut parts = fen.split_whitespace();
    let placement = parts.next().ok_or("missing placement")?;
    let turn = match parts.next().unwrap_or("w") {
        "w" => Color::White,
        "b" => Color::Black,
        x => return Err(format!("invalid turn: {x}")),
    };
    let castling_s = parts.next().unwrap_or("-");
    let ep_s = parts.next().unwrap_or("-");
    let halfmove = parts
        .next()
        .unwrap_or("0")
        .parse()
        .map_err(|_| "invalid halfmove")?;
    let fullmove = parts
        .next()
        .unwrap_or("1")
        .parse()
        .map_err(|_| "invalid fullmove")?;
    let mut squares = [None; 64];
    let ranks: Vec<&str> = placement.split('/').collect();
    if ranks.len() != 8 {
        return Err("invalid rank count".into());
    }
    for (fen_rank, row) in ranks.iter().enumerate() {
        let mut f = 0i8;
        for ch in row.chars() {
            if ch.is_ascii_digit() {
                f += ch.to_digit(10).unwrap() as i8;
            } else if let Some(piece) = piece_from_fen(ch) {
                if f >= 8 {
                    return Err("file overflow".into());
                }
                squares[idx(f, 7 - fen_rank as i8) as usize] = Some(piece);
                f += 1;
            } else {
                return Err(format!("invalid piece: {ch}"));
            }
        }
        if f != 8 {
            return Err("invalid rank width".into());
        }
    }
    let mut castling = 0u8;
    if castling_s.contains('K') {
        castling |= 1;
    }
    if castling_s.contains('Q') {
        castling |= 2;
    }
    if castling_s.contains('k') {
        castling |= 4;
    }
    if castling_s.contains('q') {
        castling |= 8;
    }
    let ep_square = if ep_s == "-" {
        None
    } else {
        Some(square_index(ep_s)?)
    };
    Ok(Board {
        squares,
        turn,
        castling,
        ep_square,
        halfmove,
        fullmove,
    })
}

pub fn board_to_fen(board: &Board) -> String {
    let mut rows = Vec::with_capacity(8);
    for r in (0..8).rev() {
        let mut row = String::new();
        let mut empty = 0;
        for f in 0..8 {
            match board.squares[(f + r * 8) as usize] {
                Some(piece) => {
                    if empty > 0 {
                        row.push_str(&empty.to_string());
                        empty = 0;
                    }
                    row.push(piece_to_fen(piece));
                }
                None => empty += 1,
            }
        }
        if empty > 0 {
            row.push_str(&empty.to_string());
        }
        rows.push(row);
    }
    let turn = if board.turn == Color::White { "w" } else { "b" };
    let mut castling = String::new();
    if board.castling & 1 != 0 {
        castling.push('K');
    }
    if board.castling & 2 != 0 {
        castling.push('Q');
    }
    if board.castling & 4 != 0 {
        castling.push('k');
    }
    if board.castling & 8 != 0 {
        castling.push('q');
    }
    if castling.is_empty() {
        castling.push('-');
    }
    let ep = board
        .ep_square
        .map(square_name)
        .unwrap_or_else(|| "-".into());
    format!(
        "{} {turn} {castling} {ep} {} {}",
        rows.join("/"),
        board.halfmove,
        board.fullmove
    )
}
