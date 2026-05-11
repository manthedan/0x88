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
