class DynamicTeam(Team):
    def __init__(self, name, id, seed, region, first_four, own, quality):
        super().__init__(name, id, seed, region, first_four, own)
        self.base_quality = quality
        self.current_quality = quality
        self.strength_updates = []
        
    def update_strength(self, opponent, expected_diff, actual_diff):
        """Update team strength based on game performance"""
        # Surprise factor - how much better/worse than expected
        surprise = actual_diff - expected_diff
        
        # Bayesian update - weight of update inversely proportional to round
        # Later rounds have more weight (we learn more about teams)
        round_weight = 0.1 * (opponent.current_quality / 0.5)  # More weight when facing stronger teams
        
        # Update current quality
        adjustment = surprise * round_weight
        self.current_quality += adjustment
        
        # Record update for analysis
        self.strength_updates.append({
            'opponent_id': opponent.id,
            'opponent_name': opponent.name,
            'expected_diff': expected_diff,
            'actual_diff': actual_diff,
            'surprise': surprise,
            'adjustment': adjustment,
            'new_quality': self.current_quality
        })
        
        return adjustment 