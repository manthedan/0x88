use crate::board::{square_name, Move, Role};

pub fn move_to_uci(m: Move) -> String {
    let promo = match m.promotion {
        Some(Role::Knight) => "n",
        Some(Role::Bishop) => "b",
        Some(Role::Rook) => "r",
        Some(Role::Queen) => "q",
        _ => "",
    };
    format!("{}{}{}", square_name(m.from), square_name(m.to), promo)
}

pub fn move_to_action_id(m: Move) -> u32 {
    let promo = match m.promotion {
        Some(Role::Knight) => 1,
        Some(Role::Bishop) => 2,
        Some(Role::Rook) => 3,
        Some(Role::Queen) => 4,
        _ => 0,
    };
    ((m.from as u32 * 64 + m.to as u32) * 5) + promo
}
