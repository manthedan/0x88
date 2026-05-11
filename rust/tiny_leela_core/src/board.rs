#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Color {
    White,
    Black,
}

impl Color {
    pub fn opposite(self) -> Self {
        match self {
            Color::White => Color::Black,
            Color::Black => Color::White,
        }
    }
    pub fn sign(self) -> f32 {
        if self == Color::White {
            1.0
        } else {
            -1.0
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Role {
    Pawn,
    Knight,
    Bishop,
    Rook,
    Queen,
    King,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Piece {
    pub color: Color,
    pub role: Role,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Board {
    pub squares: [Option<Piece>; 64],
    pub turn: Color,
    pub castling: u8, // K=1 Q=2 k=4 q=8
    pub ep_square: Option<u8>,
    pub halfmove: u16,
    pub fullmove: u16,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Move {
    pub from: u8,
    pub to: u8,
    pub promotion: Option<Role>,
}

pub const START_FEN: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
pub(crate) const KNIGHT: &[(i8, i8)] = &[
    (1, 2),
    (2, 1),
    (-1, 2),
    (-2, 1),
    (1, -2),
    (2, -1),
    (-1, -2),
    (-2, -1),
];
pub(crate) const KING: &[(i8, i8)] = &[
    (1, 1),
    (1, 0),
    (1, -1),
    (0, 1),
    (0, -1),
    (-1, 1),
    (-1, 0),
    (-1, -1),
];
pub(crate) const BISHOP: &[(i8, i8)] = &[(1, 1), (1, -1), (-1, 1), (-1, -1)];
pub(crate) const ROOK: &[(i8, i8)] = &[(1, 0), (-1, 0), (0, 1), (0, -1)];

pub(crate) fn file(sq: u8) -> i8 {
    (sq % 8) as i8
}
pub(crate) fn rank(sq: u8) -> i8 {
    (sq / 8) as i8
}
pub(crate) fn on(f: i8, r: i8) -> bool {
    (0..8).contains(&f) && (0..8).contains(&r)
}
pub(crate) fn idx(f: i8, r: i8) -> u8 {
    (f + r * 8) as u8
}

pub fn square_index(name: &str) -> Result<u8, String> {
    let b = name.as_bytes();
    if b.len() != 2 || !(b'a'..=b'h').contains(&b[0]) || !(b'1'..=b'8').contains(&b[1]) {
        return Err(format!("invalid square: {name}"));
    }
    Ok((b[0] - b'a') + (b[1] - b'1') * 8)
}

pub fn square_name(sq: u8) -> String {
    let f = (b'a' + sq % 8) as char;
    let r = (b'1' + sq / 8) as char;
    format!("{f}{r}")
}

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
