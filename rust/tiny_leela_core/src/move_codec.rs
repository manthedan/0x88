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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uci_and_action_id_include_promotion() {
        let m = Move {
            from: 48,
            to: 56,
            promotion: Some(Role::Queen),
        };
        assert_eq!(move_to_uci(m), "a7a8q");
        assert_eq!(move_to_action_id(m), ((48u32 * 64 + 56u32) * 5) + 4);
    }
}
