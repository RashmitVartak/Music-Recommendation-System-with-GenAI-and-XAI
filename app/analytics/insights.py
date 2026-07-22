class RecommendationInsights:

    @staticmethod
    def overall(score):
        if score >= 80:
            return "Excellent overall recommendation diversity with balanced artist, year and popularity coverage."
        elif score >= 60:
            return "Good recommendation diversity with slight concentration in some attributes."
        elif score >= 40:
            return "Excellent overall recommendation diversity with balanced artist,\n year and popularity coverage."

        return "Low diversity. Recommendations are concentrated around similar songs."

    @staticmethod
    def artist(score):
        if score >= 80:
            return "Recommendations include a wide variety of artists."
        elif score >= 60:
            return "Most recommendations come from different artists."
        elif score >= 40:
            return "Several artists are repeated in the recommendation list."

        return "Recommendations are dominated by only a few artists."

    @staticmethod
    def year(score):
        if score >= 80:
            return "Songs span multiple release years."
        elif score >= 60:
            return "Recommendations cover a reasonable range of years."
        elif score >= 40:
            return "Recommendations are somewhat concentrated in one era."

        return "Recommendations are heavily biased toward a specific time period."

    @staticmethod
    def popularity(score):
        if score >= 80:
            return "Excellent balance between mainstream and niche music."
        elif score >= 60:
            return "Good mix of popular and lesser-known tracks."
        elif score >= 40:
            return "Recommendations lean toward either popular or niche songs."

        return "Recommendations lack popularity diversity."