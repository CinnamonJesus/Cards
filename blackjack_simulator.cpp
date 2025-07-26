/*
 * Blackjack Simulator and Advisor
 *
 * This C++ program is a direct reinterpretation of a Python‑based
 * blackjack simulator originally provided in a PyQt6 application.  The
 * goals of the rewrite are threefold: (1) preserve the statistical
 * integrity of the original algorithms, (2) expose well‑structured C++
 * classes that separate concerns such as card counting, shoe
 * management, Bayesian prediction and Monte‑Carlo simulation, and
 * (3) provide an entry point for future GUI integration using a
 * framework such as Qt.  To this end, the code below is divided
 * logically into several classes with clear responsibilities.  All
 * stateful objects encapsulate their data and perform bounds
 * checking where appropriate.
 *
 * The simulation routines rely on the C++17 standard library only.
 * Randomness is supplied by std::mt19937 and std::uniform_int_distribution.
 * Multi‑threading is optional; for portability the FastSimulator class
 * exposes a threadCount parameter but falls back to sequential
 * execution if std::thread::hardware_concurrency reports one.
 */

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <exception>
#include <functional>
#include <future>
#include <iostream>
#include <map>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include <unordered_set>
#include <thread>
#include <array>
#include <cstdio>

/* Forward declarations */
class Shoe;

/**
 * Helper functions for manipulating card codes.  Card codes are
 * represented as strings composed of a rank followed by a single
 * character suit.  The rank portion may be one or two characters
 * long ("10"), and the suit is one of 'S', 'H', 'D' or 'C'.
 */
namespace CardUtil {
    inline std::string rankFromCode(const std::string &code) {
        // Extract the rank portion of a card code.  Note that "10" is
        // represented explicitly; all other ranks are a single
        // character at the front of the string.
        if (code.size() >= 2 && code[0] == '1' && code[1] == '0') {
            return "10";
        }
        // rank is the first character
        return std::string(1, code[0]);
    }
}

/**
 * ZenCount implements the Zen card counting system.  Each rank is
 * associated with an integer weight and a running count is updated
 * as cards are removed from or restored to the shoe.  The class
 * exposes methods for updating, undoing updates and computing the
 * true count based on the number of decks remaining.
 */
class ZenCount {
public:
    ZenCount() : runningCount(0) {}

    void update(const std::string &cardCode) {
        auto rank = CardUtil::rankFromCode(cardCode);
        auto it = values.find(rank);
        if (it != values.end()) {
            runningCount += it->second;
        }
    }

    void undo(const std::string &cardCode) {
        auto rank = CardUtil::rankFromCode(cardCode);
        auto it = values.find(rank);
        if (it != values.end()) {
            runningCount -= it->second;
        }
    }

    void reset() {
        runningCount = 0;
    }

    double trueCount(double decksRemaining) const {
        if (decksRemaining <= 0.0) return 0.0;
        return static_cast<double>(runningCount) / decksRemaining;
    }

    int getRunningCount() const { return runningCount; }

private:
    int runningCount;
    static const std::unordered_map<std::string, int> values;
};

const std::unordered_map<std::string, int> ZenCount::values = {
    {"2", +1}, {"3", +1}, {"4", +2}, {"5", +2}, {"6", +2},
    {"7", +1}, {"8", 0},  {"9", 0},  {"10", -2}, {"J", -2},
    {"Q", -2}, {"K", -2}, {"A", -1}
};

/**
 * WongHalves implements the Wong Halves counting system using
 * fractional weights.  Behaviour mirrors ZenCount but uses double
 * values for the running count and weights.
 */
class WongHalves {
public:
    WongHalves() : runningCount(0.0) {}

    void update(const std::string &cardCode) {
        auto rank = CardUtil::rankFromCode(cardCode);
        auto it = values.find(rank);
        if (it != values.end()) {
            runningCount += it->second;
        }
    }

    void undo(const std::string &cardCode) {
        auto rank = CardUtil::rankFromCode(cardCode);
        auto it = values.find(rank);
        if (it != values.end()) {
            runningCount -= it->second;
        }
    }

    void reset() {
        runningCount = 0.0;
    }

    double trueCount(double decksRemaining) const {
        if (decksRemaining <= 0.0) return 0.0;
        return runningCount / decksRemaining;
    }

    double getRunningCount() const { return runningCount; }

private:
    double runningCount;
    static const std::unordered_map<std::string, double> values;
};

const std::unordered_map<std::string, double> WongHalves::values = {
    {"2", +0.5}, {"3", +1.0}, {"4", +1.0}, {"5", +1.5}, {"6", +1.0},
    {"7", +0.5}, {"8", 0.0}, {"9", -0.5}, {"10", -1.0}, {"J", -1.0},
    {"Q", -1.0}, {"K", -1.0}, {"A", -1.0}
};

/**
 * The Shoe class manages one or more decks of cards.  Internally,
 * cards are stored as a map from card code (e.g. "AS" for Ace of
 * spades) to the number of such cards remaining.  Removing and
 * restoring cards adjust the running penetration.  The class throws
 * a std::runtime_error if a removal is attempted for a card no
 * longer available.
 */
class Shoe {
public:
    explicit Shoe(int decks = 8) : decks(decks) {
        resetShoe();
    }

    void resetShoe() {
        static const std::vector<std::string> ranks = {
            "A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"};
        static const std::vector<char> suits = {'S', 'H', 'D', 'C'};
        cards.clear();
        for (const auto &r : ranks) {
            for (const auto &s : suits) {
                std::string code = r + std::string(1, s);
                cards[code] = decks;
            }
        }
        totalCards = 52 * decks;
        penetration = 0.0;
    }

    /**
     * Remove a card from the shoe.  Throws if the card is not
     * available.  After removal, updates the total and penetration.
     */
    void removeCard(const std::string &cardCode) {
        auto it = cards.find(cardCode);
        if (it == cards.end() || it->second <= 0) {
            throw std::runtime_error("Card " + cardCode + " is not available to remove.");
        }
        it->second -= 1;
        totalCards -= 1;
        penetration = (static_cast<double>(52 * decks - totalCards)) / (52.0 * decks);
    }

    /**
     * Restore a card back into the shoe.  If the card is not known,
     * it will be added; otherwise the count is incremented up to the
     * original number of decks.
     */
    void restoreCard(const std::string &cardCode) {
        auto it = cards.find(cardCode);
        if (it == cards.end()) {
            // unknown card, assume one instance
            cards[cardCode] = 1;
        } else {
            if (it->second < decks) {
                it->second += 1;
            }
        }
        totalCards += 1;
        penetration = (static_cast<double>(52 * decks - totalCards)) / (52.0 * decks);
    }

    /**
     * Compute the number of decks remaining as a floating point
     * quantity.  Since partial decks are possible, this value is
     * totalCards divided by 52.
     */
    double decksRemaining() const {
        return totalCards / 52.0;
    }

    /**
     * Return the penetration (fraction of the shoe that has been dealt).
     */
    double getPenetration() const {
        return penetration;
    }

    /**
     * Retrieve a map of all cards still present in the shoe.  Only
     * entries with a positive count are returned.
     */
    std::map<std::string, int> getRemainingCards() const {
        std::map<std::string, int> remaining;
        for (const auto &kv : cards) {
            if (kv.second > 0) remaining[kv.first] = kv.second;
        }
        return remaining;
    }

    /**
     * Expose underlying counts for simulation.  Use with caution.
     */
    const std::map<std::string, int> &getCards() const { return cards; }

    int getTotalCards() const { return totalCards; }

private:
    int decks;
    std::map<std::string, int> cards;
    int totalCards;
    double penetration;
};

/**
 * The SideBets namespace encapsulates utility functions related to
 * blackjack side bets.  Payout schedules are static maps and
 * evaluate* functions return string keys identifying the bet hit or
 * std::nullopt if no bet is made.  Expected value functions accept
 * probability distributions and accumulate winnings accordingly.
 */
namespace SideBets {
    // Payout tables
    static const std::unordered_map<std::string, int> PAYOUT_21PLUS3 = {
        {"flush", 5}, {"straight", 10}, {"three_kind", 30}, {"straight_flush", 40}, {"suited_trips", 100}
    };
    static const std::unordered_map<std::string, int> PAYOUT_PAIR = {
        {"pair", 8}, {"suited_pair", 25}
    };
    static const std::unordered_map<std::string, int> PAYOUT_BUST = {
        {"3", 1}, {"4", 2}, {"5", 9}, {"6", 50}, {"7", 100}, {"8+", 250}
    };
    static const std::unordered_map<std::string, int> PAYOUT_HOT3 = {
        {"777", 100}, {"21suited", 20}, {"21", 10}, {"20suited", 4}, {"20", 2}, {"19", 1}
    };

    // Evaluate 21+3 bet for a given two player cards and dealer upcard
    inline std::string evaluate21Plus3(const std::vector<std::string> &playerCards,
                                       const std::string &dealerUpcard) {
        if (playerCards.size() < 2 || dealerUpcard.empty()) return {};
        std::vector<std::string> cards = playerCards;
        cards.push_back(dealerUpcard);
        // Extract ranks and suits
        std::vector<std::string> ranks;
        std::vector<char> suits;
        std::vector<int> rankValues;
        static const std::unordered_map<std::string, int> rankOrder = {
            {"A", 1}, {"2", 2}, {"3", 3}, {"4", 4}, {"5", 5}, {"6", 6},
            {"7", 7}, {"8", 8}, {"9", 9}, {"10", 10}, {"J", 11}, {"Q", 12}, {"K", 13}
        };
        for (const auto &c : cards) {
            std::string rank = CardUtil::rankFromCode(c);
            char suit = c.back();
            ranks.push_back(rank);
            suits.push_back(suit);
            auto it = rankOrder.find(rank);
            rankValues.push_back(it != rankOrder.end() ? it->second : 0);
        }
        std::sort(rankValues.begin(), rankValues.end());
        bool flush = std::unordered_set<char>(suits.begin(), suits.end()).size() == 1;
        bool threeKind = std::unordered_set<std::string>(ranks.begin(), ranks.end()).size() == 1;
        bool straight = false;
        // check normal straight, including Ace low and high (A,2,3) and (Q,K,A)
        if (rankValues[1] == rankValues[0] + 1 && rankValues[2] == rankValues[1] + 1) straight = true;
        std::vector<int> low = {1, 2, 3};
        std::vector<int> high = {1, 12, 13};
        if (rankValues == low || rankValues == high) straight = true;

        if (threeKind) {
            return flush ? "suited_trips" : "three_kind";
        }
        if (flush && straight) return "straight_flush";
        if (flush) return "flush";
        if (straight) return "straight";
        return {};
    }

    // Evaluate perfect pair side bet
    inline std::string evaluatePair(const std::vector<std::string> &playerCards) {
        if (playerCards.size() < 2) return {};
        const auto &c1 = playerCards[0];
        const auto &c2 = playerCards[1];
        std::string r1 = CardUtil::rankFromCode(c1);
        std::string r2 = CardUtil::rankFromCode(c2);
        if (r1 == r2) {
            return (c1.back() == c2.back()) ? "suited_pair" : "pair";
        }
        return {};
    }

    // Evaluate Bust‑O‑Rama side bet.  The argument is the number of
    // cards the dealer takes before busting.  If numCardsToBust is
    // greater than or equal to 8 the payout is keyed by "8+".
    inline int evaluateBust(int numCardsToBust) {
        if (numCardsToBust >= 8) {
            return PAYOUT_BUST.at("8+");
        }
        auto it = PAYOUT_BUST.find(std::to_string(numCardsToBust));
        return it != PAYOUT_BUST.end() ? it->second : 0;
    }

    // Evaluate Hot3 side bet
    inline std::string evaluateHot3(const std::vector<std::string> &playerCards,
                                    const std::string &dealerUpcard) {
        if (playerCards.size() != 2 || dealerUpcard.empty()) return {};
        std::vector<std::string> cards = playerCards;
        cards.push_back(dealerUpcard);
        std::vector<char> suits;
        std::vector<std::string> ranks;
        int total = 0;
        int aceCount = 0;
        for (const auto &c : cards) {
            char suit = c.back();
            suits.push_back(suit);
            std::string r = CardUtil::rankFromCode(c);
            ranks.push_back(r);
            if (r == "J" || r == "Q" || r == "K" || r == "10") {
                total += 10;
            } else if (r == "A") {
                total += 11;
                aceCount++;
            } else {
                total += std::stoi(r);
            }
        }
        while (total > 21 && aceCount > 0) {
            total -= 10;
            aceCount--;
        }
        if (total == 21) {
            bool allSevens = std::all_of(ranks.begin(), ranks.end(), [](const std::string &r){ return r == "7"; });
            if (allSevens) return "777";
            return (std::unordered_set<char>(suits.begin(), suits.end()).size() == 1) ? "21suited" : "21";
        } else if (total == 20) {
            return (std::unordered_set<char>(suits.begin(), suits.end()).size() == 1) ? "20suited" : "20";
        } else if (total == 19) {
            return "19";
        }
        return {};
    }

    // Expected value computations.  Each function accepts a probability
    // distribution keyed by outcome and multiplies by the relevant
    // payout table.
    inline double expectedValue21Plus3(const std::map<std::string, double> &probs) {
        double ev = 0.0;
        for (const auto &kv : probs) {
            auto it = PAYOUT_21PLUS3.find(kv.first);
            if (it != PAYOUT_21PLUS3.end()) ev += kv.second * it->second;
        }
        return ev;
    }
    inline double expectedValuePair(const std::map<std::string, double> &probs) {
        double ev = 0.0;
        for (const auto &kv : probs) {
            auto it = PAYOUT_PAIR.find(kv.first);
            if (it != PAYOUT_PAIR.end()) ev += kv.second * it->second;
        }
        return ev;
    }
    inline double expectedValueBust(const std::map<int, double> &probs) {
        double ev = 0.0;
        for (const auto &kv : probs) {
            if (kv.first >= 8) {
                ev += kv.second * PAYOUT_BUST.at("8+");
            } else {
                auto it = PAYOUT_BUST.find(std::to_string(kv.first));
                if (it != PAYOUT_BUST.end()) ev += kv.second * it->second;
            }
        }
        return ev;
    }
    inline double expectedValueHot3(const std::map<std::string, double> &probs) {
        double ev = 0.0;
        for (const auto &kv : probs) {
            auto it = PAYOUT_HOT3.find(kv.first);
            if (it != PAYOUT_HOT3.end()) ev += kv.second * it->second;
        }
        return ev;
    }
} // namespace SideBets

/**
 * BayesianPredictor offers static functions that compute probabilistic
 * summaries over the remaining shoe.  It includes next card
 * probabilities, probability of specific sequences, heatmaps, group
 * probabilities and a simple entropy–based confidence score.  The
 * interface uses std::map and std::vector extensively to remain
 * explicit and straightforward.
 */
class BayesianPredictor {
public:
    /**
     * Compute the top N most likely next cards from the shoe.  The
     * resulting vector is sorted descending by probability.  Only
     * cards with a positive count are considered.
     */
    static std::vector<std::pair<std::string, double>> nextCardProbabilities(
        const std::map<std::string, int> &shoeCards,
        std::size_t N = 5) {
        std::vector<std::pair<std::string, double>> out;
        int total = 0;
        for (const auto &kv : shoeCards) total += kv.second;
        if (total <= 0) return out;
        for (const auto &kv : shoeCards) {
            if (kv.second > 0) {
                out.emplace_back(kv.first, static_cast<double>(kv.second) / total);
            }
        }
        std::sort(out.begin(), out.end(), [](auto &a, auto &b){ return a.second > b.second; });
        if (out.size() > N) out.resize(N);
        return out;
    }

    /**
     * Given an unordered shoe and a desired sequence of card codes,
     * compute the exact probability of that sequence occurring.
     * Probability is computed without replacement.  If any card is
     * absent the probability is zero.
     */
    static double probabilityOfSequence(std::map<std::string, int> shoeCards,
                                        const std::vector<std::string> &targetSequence) {
        long double logp = 0.0;
        int totalCards = 0;
        for (const auto &kv : shoeCards) totalCards += kv.second;
        if (totalCards == 0) return 0.0;
        for (const auto &card : targetSequence) {
            int count = shoeCards[card];
            if (count <= 0) return 0.0;
            logp += std::log(static_cast<long double>(count) / totalCards);
            shoeCards[card] -= 1;
            totalCards -= 1;
        }
        return std::exp(logp);
    }

    /**
     * Return a full heatmap of card frequencies in the shoe.  Only
     * nonzero counts are included.  The values sum to 1.
     */
    static std::map<std::string, double> fullCardHeatmap(const std::map<std::string, int> &shoeCards) {
        std::map<std::string, double> result;
        int total = 0;
        for (const auto &kv : shoeCards) total += kv.second;
        if (total <= 0) return result;
        for (const auto &kv : shoeCards) {
            if (kv.second > 0) result[kv.first] = static_cast<double>(kv.second) / total;
        }
        return result;
    }

    /**
     * Compute probabilities of drawing a small, mid or big card.  The
     * definitions mirror the original Python implementation: small
     * cards are 2–6, mid are 7–9 and big are 10–A.  The return
     * map's keys are "small", "mid" and "big".
     */
    static std::map<std::string, double> cardGroupProbabilities(const std::map<std::string, int> &shoeCards) {
        static const std::vector<std::string> small = {"2", "3", "4", "5", "6"};
        static const std::vector<std::string> mid   = {"7", "8", "9"};
        static const std::vector<std::string> big   = {"10", "J", "Q", "K", "A"};
        std::map<std::string, double> counts = {{"small", 0.0}, {"mid", 0.0}, {"big", 0.0}};
        double total = 0.0;
        for (const auto &kv : shoeCards) total += kv.second;
        if (total <= 0.0) return counts;
        for (const auto &kv : shoeCards) {
            std::string rank = CardUtil::rankFromCode(kv.first);
            double count = kv.second;
            if (std::find(small.begin(), small.end(), rank) != small.end()) {
                counts["small"] += count;
            } else if (std::find(mid.begin(), mid.end(), rank) != mid.end()) {
                counts["mid"] += count;
            } else {
                counts["big"] += count;
            }
        }
        for (auto &kv : counts) {
            kv.second /= total;
        }
        return counts;
    }

    /**
     * Internal helper used by bayesConfidenceScore to compute base‑2
     * entropy of a vector of counts.  The counts vector should
     * contain only positive values.  A zero total yields zero
     * entropy.
     */
    static double entropyFromCounts(const std::vector<int> &values) {
        double total = 0.0;
        for (int v : values) total += v;
        if (total <= 0.0) return 0.0;
        double ent = 0.0;
        for (int v : values) {
            if (v > 0) {
                double p = v / total;
                ent -= p * std::log2(p);
            }
        }
        return ent;
    }

    /**
     * Compute a confidence score based on the entropy of the card
     * distribution.  The maximum entropy occurs when all cards are
     * equally likely; confidence is defined as 1 – (entropy /
     * maxEntropy).  A higher score indicates the remaining shoe is
     * depleted and predictions are more reliable.
     */
    static double bayesConfidenceScore(const std::map<std::string, int> &shoeCards) {
        std::vector<int> counts;
        counts.reserve(shoeCards.size());
        for (const auto &kv : shoeCards) if (kv.second > 0) counts.push_back(kv.second);
        double entropy = entropyFromCounts(counts);
        double maxEntropy = 0.0;
        if (!shoeCards.empty()) {
            maxEntropy = std::log2(static_cast<double>(shoeCards.size()));
        } else {
            maxEntropy = 1.0;
        }
        return maxEntropy > 0.0 ? 1.0 - (entropy / maxEntropy) : 0.0;
    }

    /**
     * Simulate dealer outcomes given a particular upcard and the
     * composition of the shoe.  Returns a map keyed by total (or
     * "bust") to probability.  The algorithm draws without
     * replacement and uses standard blackjack dealer rules: draw
     * until reaching at least 17, but stand on soft 17.
     */
    static std::map<std::string, double>
    dealerTotalProbabilities(const std::string &upcard,
                             const std::map<std::string, int> &shoeCards,
                             std::size_t simulations = 10000) {
        std::map<std::string, double> counts;
        // Flatten the shoe into a vector of card codes for easy sampling
        std::vector<std::string> flat;
        flat.reserve(52 * 8); // an upper bound; actual size may be smaller
        for (const auto &kv : shoeCards) {
            for (int i = 0; i < kv.second; ++i) flat.push_back(kv.first);
        }
        if (std::find(flat.begin(), flat.end(), upcard) == flat.end()) {
            return counts;
        }
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dist;
        for (std::size_t i = 0; i < simulations; ++i) {
            // make a copy of the flat shoe
            std::vector<std::string> local = flat;
            // remove the upcard from local
            auto it = std::find(local.begin(), local.end(), upcard);
            if (it != local.end()) local.erase(it);
            std::vector<std::string> hand;
            hand.push_back(upcard);
            while (true) {
                // compute dealer total and aces
                int total = 0;
                int aces = 0;
                for (const auto &card : hand) {
                    std::string rank = CardUtil::rankFromCode(card);
                    int val;
                    if (rank == "A") val = 11;
                    else if (rank == "J" || rank == "Q" || rank == "K" || rank == "10") val = 10;
                    else val = std::stoi(rank);
                    total += val;
                    if (rank == "A") aces++;
                }
                while (total > 21 && aces > 0) {
                    total -= 10;
                    aces--;
                }
                // decide whether to hit or stand
                bool soft = false;
                // check for soft
                int testTotal = 0;
                int testAces = 0;
                for (const auto &card : hand) {
                    std::string rank = CardUtil::rankFromCode(card);
                    int val;
                    if (rank == "A") val = 11;
                    else if (rank == "J" || rank == "Q" || rank == "K" || rank == "10") val = 10;
                    else val = std::stoi(rank);
                    testTotal += val;
                    if (rank == "A") testAces++;
                }
                while (testTotal > 21 && testAces > 0) {
                    testTotal -= 10;
                    testAces--;
                }
                soft = (testAces > 0 && testTotal <= 21);
                if (total < 17) {
                    // draw a card
                    std::uniform_int_distribution<> d(0, static_cast<int>(local.size()) - 1);
                    int idx = d(gen);
                    std::string drawn = local[idx];
                    local.erase(local.begin() + idx);
                    hand.push_back(drawn);
                } else if (total == 17 && soft) {
                    // stand on soft 17 in this game; break
                    break;
                } else {
                    break;
                }
            }
            // evaluate final hand
            int total = 0;
            int aces = 0;
            for (const auto &card : hand) {
                std::string rank = CardUtil::rankFromCode(card);
                int val;
                if (rank == "A") val = 11;
                else if (rank == "J" || rank == "Q" || rank == "K" || rank == "10") val = 10;
                else val = std::stoi(rank);
                total += val;
                if (rank == "A") aces++;
            }
            while (total > 21 && aces > 0) {
                total -= 10;
                aces--;
            }
            if (total > 21) {
                counts["bust"] += 1.0;
            } else {
                counts[std::to_string(total)] += 1.0;
            }
        }
        // normalize
        std::map<std::string, double> probs;
        for (const auto &kv : counts) {
            probs[kv.first] = kv.second / static_cast<double>(simulations);
        }
        return probs;
    }
};

/**
 * DecisionAdvisor encapsulates the strategy index plays and true
 * count conditions from the original Python function.  The static
 * recommendAction function takes the player's hand, the dealer
 * upcard and the current true count and returns a string
 * recommendation.  Configuration is passed via an optional map;
 * reasonable defaults are provided if omitted.
 */
class DecisionAdvisor {
public:
    struct Config {
        double insuranceThreshold = 3.0;
        std::unordered_map<std::string, int> indexPlays = {
            {"16v10", 0}, {"15v10", 4}, {"13v2", -1}, {"12v2", 3}, {"12v3", 3}
        };
    };

    static std::string recommendAction(const std::vector<std::string> &playerHand,
                                       const std::string &dealerUpcard,
                                       double trueCount,
                                       const Config &cfg = Config()) {
        if (playerHand.size() < 2 || dealerUpcard.empty()) {
            return "Awaiting full input";
        }
        auto cardValue = [](const std::string &rank) {
            if (rank == "A") return 11;
            if (rank == "J" || rank == "Q" || rank == "K" || rank == "10") return 10;
            return std::stoi(rank);
        };
        auto rank = [](const std::string &code) {
            return CardUtil::rankFromCode(code);
        };
        std::string p1 = rank(playerHand[0]);
        std::string p2 = rank(playerHand[1]);
        std::string dealer = rank(dealerUpcard);
        int v1 = cardValue(p1);
        int v2 = cardValue(p2);
        int playerTotal = v1 + v2;
        int dealerVal = cardValue(dealer);
        bool soft = ((p1 == "A" || p2 == "A") && playerTotal <= 21);
        bool pair = (p1 == p2);
        // Insurance check
        if (dealer == "A") {
            if (trueCount >= cfg.insuranceThreshold) return "Insurance: Take it";
            else return "Insurance: Decline";
        }
        if (pair) {
            if (p1 == "A") return "Split Aces";
            if (p1 == "8") return "Split 8s";
            if (p1 == "9") return (dealerVal == 7 || dealerVal == 10 || dealerVal == 11) ? "Stand" : "Split 9s";
            if (p1 == "7" && dealerVal <= 7) return "Split 7s";
            if (p1 == "6" && dealerVal <= 6) return "Split 6s";
            if (p1 == "4" && (dealerVal == 5 || dealerVal == 6)) return "Split 4s";
            if (p1 == "3" && dealerVal <= 7) return "Split 3s";
            if (p1 == "2" && dealerVal <= 7) return "Split 2s";
            return "Don't Split";
        }
        if (soft) {
            if (playerTotal >= 19) return "Stand";
            if (playerTotal == 18) {
                if (dealerVal == 2 || dealerVal == 7 || dealerVal == 8) return "Stand";
                if (dealerVal >= 3 && dealerVal <= 6) return "Double";
                return "Hit";
            }
            if (playerTotal == 17 && (dealerVal >= 3 && dealerVal <= 6)) return "Double";
            return "Hit";
        }
        if (playerTotal >= 17) return "Stand";
        if (playerTotal == 16 && dealerVal == 10) {
            return trueCount >= cfg.indexPlays.at("16v10") ? "Stand" : "Hit";
        }
        if (playerTotal == 15 && dealerVal == 10) {
            return trueCount >= cfg.indexPlays.at("15v10") ? "Stand" : "Hit";
        }
        if (playerTotal == 13 && dealerVal == 2) {
            return trueCount >= cfg.indexPlays.at("13v2") ? "Stand" : "Hit";
        }
        if (playerTotal == 12) {
            if (dealerVal == 2) {
                return trueCount >= cfg.indexPlays.at("12v2") ? "Stand" : "Hit";
            }
            if (dealerVal == 3) {
                return trueCount >= cfg.indexPlays.at("12v3") ? "Stand" : "Hit";
            }
            if (dealerVal >= 4 && dealerVal <= 6) return "Stand";
            return "Hit";
        }
        if (playerTotal == 11) return "Double";
        if (playerTotal == 10) return (dealerVal <= 9) ? "Double" : "Hit";
        if (playerTotal == 9) return (dealerVal >= 3 && dealerVal <= 6) ? "Double" : "Hit";
        return "Hit";
    }
};

/**
 * FastSimulator is responsible for running large batches of blackjack
 * hands to estimate expected values for the main game and several
 * side bets.  It uses a flat array representation of the shoe for
 * speed and optionally spawns multiple threads to distribute
 * simulation workload.  Methods that modify shared totals use
 * thread‑local accumulators to avoid contention.
 */
class FastSimulator {
public:
    explicit FastSimulator(const Shoe &shoe) {
        // encode the shoe into a 52‑element array for faster indexing
        encodeShoe(shoe.getCards());
    }

    /**
     * Run simulations in parallel and return expected values for
     * main EV, 21+3, pair, Hot3 and bust bets.  batchCount controls
     * how many threads are spawned.  Each thread runs the given
     * number of rounds.
     */
    std::map<std::string, double> runBatch(unsigned batchCount = 4, unsigned rounds = 100000) {
        // Determine how many hardware threads we can use
        unsigned hardware = std::thread::hardware_concurrency();
        unsigned threadCount = std::min(batchCount, hardware == 0 ? 1 : hardware);
        // If threadCount is zero fallback to one
        if (threadCount == 0) threadCount = 1;
        // Launch threads
        std::vector<std::future<std::array<double,5>>> futures;
        for (unsigned t = 0; t < threadCount; ++t) {
            futures.emplace_back(std::async(std::launch::async, [this, rounds]() {
                return simulateChunk(rounds);
            }));
        }
        // Aggregate results
        double sumMain = 0.0;
        double sumBust = 0.0;
        double sum21   = 0.0;
        double sumPair = 0.0;
        double sumHot3 = 0.0;
        for (auto &f : futures) {
            auto res = f.get();
            sumMain += res[0];
            sumBust += res[1];
            sum21   += res[2];
            sumPair += res[3];
            sumHot3 += res[4];
        }
        double totalRounds = static_cast<double>(threadCount) * rounds;
        std::map<std::string, double> evs;
        evs["main_ev"]  = sumMain  / totalRounds;
        evs["bust_ev"]  = sumBust  / totalRounds;
        evs["21+3_ev"] = sum21   / totalRounds;
        evs["pair_ev"] = sumPair / totalRounds;
        evs["hot3_ev"] = sumHot3 / totalRounds;
        return evs;
    }

private:
    // flat representation of shoe counts.  Index assignment: ranks
    // ordered by A,2,3,...,K and suits by S,H,D,C; this yields 52
    // elements.  The encodeShoe function populates this array.
    std::array<int, 52> shoeBuffer{};

    void encodeShoe(const std::map<std::string, int> &shoeDict) {
        // build mapping once
        static const std::map<std::string, int> cardIndex = []() {
            std::map<std::string, int> m;
            std::vector<std::string> ranks = {
                "A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"
            };
            std::vector<char> suits = {'S','H','D','C'};
            int idx = 0;
            for (const auto &r : ranks) {
                for (const auto &s : suits) {
                    m[r + std::string(1, s)] = idx++;
                }
            }
            return m;
        }();
        shoeBuffer.fill(0);
        for (const auto &kv : shoeDict) {
            auto it = cardIndex.find(kv.first);
            if (it != cardIndex.end()) shoeBuffer[it->second] = kv.second;
        }
    }

    // decode an index back into a card code.  Uses the same
    // ordering as cardIndex above.
    static std::string decodeCard(int idx) {
        static const std::vector<std::string> indexCard = []() {
            std::vector<std::string> out;
            std::vector<std::string> ranks = {
                "A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"
            };
            std::vector<char> suits = {'S','H','D','C'};
            for (const auto &r : ranks) {
                for (const auto &s : suits) {
                    out.push_back(r + std::string(1, s));
                }
            }
            return out;
        }();
        return indexCard[idx];
    }

    // Random card drawing from shoeBuffer
    int drawCard(std::array<int,52> &shoe, std::mt19937 &gen) {
        int total = 0;
        for (int c : shoe) total += c;
        if (total <= 0) return -1;
        std::uniform_int_distribution<> dist(0, total - 1);
        int choice = dist(gen);
        int acc = 0;
        for (int i = 0; i < 52; ++i) {
            acc += shoe[i];
            if (acc > choice) {
                shoe[i] -= 1;
                return i;
            }
        }
        return -1;
    }

    // Convert rank index into base value (A is 11, 2 is 2, ..., J/Q/K/10 are 10)
    int cardValue(int rankIndex) {
        if (rankIndex == 0) return 11; // Ace
        if (rankIndex >= 1 && rankIndex <= 9) return rankIndex + 1; // 2..10
        return 10; // J,Q,K
    }

    int getCardRank(int idx) {
        return idx / 4;
    }

    // Evaluate a hand's total, taking soft Aces into account
    int handValue(const std::vector<int> &hand) {
        int total = 0;
        int aces = 0;
        for (int idx : hand) {
            int val = cardValue(getCardRank(idx));
            total += val;
            if (val == 11) aces++;
        }
        while (total > 21 && aces > 0) {
            total -= 10;
            aces--;
        }
        return total;
    }

    bool isSoft(const std::vector<int> &hand) {
        int total = 0;
        int aces = 0;
        for (int idx : hand) {
            int val = cardValue(getCardRank(idx));
            total += val;
            if (val == 11) aces++;
        }
        while (total > 21 && aces > 0) {
            total -= 10;
            aces--;
        }
        return aces > 0 && total <= 21;
    }

    // Basic strategy action: returns 'h' (hit), 's' (stand) or 'd' (double)
    char basicAction(int total, bool soft, int dealerVal) {
        if (soft) {
            if (total == 13 || total == 14) return (dealerVal == 5 || dealerVal == 6) ? 'd' : 'h';
            if (total == 15 || total == 16) return (dealerVal >= 4 && dealerVal <= 6) ? 'd' : 'h';
            if (total == 17) return (dealerVal >= 3 && dealerVal <= 6) ? 'd' : 'h';
            if (total == 18) {
                if (dealerVal >= 3 && dealerVal <= 6) return 'd';
                if (dealerVal == 2 || dealerVal == 7 || dealerVal == 8) return 's';
                return 'h';
            }
            return 's';
        }
        if (total <= 8) return 'h';
        if (total == 9) return (dealerVal >= 3 && dealerVal <= 6) ? 'd' : 'h';
        if (total == 10) return (dealerVal <= 9) ? 'd' : 'h';
        if (total == 11) return (dealerVal <= 10) ? 'd' : 'h';
        if (total == 12) return (dealerVal >= 4 && dealerVal <= 6) ? 's' : 'h';
        if (total >= 13 && total <= 16) return (dealerVal <= 6) ? 's' : 'h';
        return 's';
    }

    // Play the player's hand according to basic strategy.  Returns final
    // total and bet multiplier (1.0 or 2.0 if doubled).  The shoe is
    // modified in place.
    std::pair<int,double> playPlayer(std::vector<int> &hand, int dealerVal, std::array<int,52> &shoe, std::mt19937 &gen) {
        int total = handValue(hand);
        bool soft = isSoft(hand);
        double bet = 1.0;
        if (hand.size() == 2) {
            char act = basicAction(total, soft, dealerVal);
            if (act == 'd') {
                int card = drawCard(shoe, gen);
                if (card >= 0) {
                    hand.push_back(card);
                    bet = 2.0;
                }
                return { handValue(hand), bet };
            }
        }
        while (true) {
            char act = basicAction(handValue(hand), isSoft(hand), dealerVal);
            if (act == 'h') {
                int card = drawCard(shoe, gen);
                if (card >= 0) {
                    hand.push_back(card);
                    if (handValue(hand) > 21) break;
                } else {
                    break;
                }
            } else {
                break;
            }
        }
        return { handValue(hand), bet };
    }

    // Play the dealer hand according to house rules.  Returns final
    // total and number of cards drawn.  The shoe is modified in place.
    std::pair<int,int> playDealer(std::vector<int> &dealer, std::array<int,52> &shoe, std::mt19937 &gen) {
        while (true) {
            int total = handValue(dealer);
            if (total < 17) {
                int card = drawCard(shoe, gen);
                if (card >= 0) {
                    dealer.push_back(card);
                } else {
                    break;
                }
            } else if (total == 17 && isSoft(dealer)) {
                // In this implementation, soft 17 stands
                break;
            } else {
                break;
            }
        }
        return { handValue(dealer), static_cast<int>(dealer.size()) };
    }

    // Simulate a chunk of rounds, returning accumulators: [mainEV, bustWins,
    // 21+3 wins, pair wins, hot3 wins].
    std::array<double,5> simulateChunk(unsigned rounds) {
        // thread local random generator
        std::random_device rd;
        std::mt19937 gen(rd());
        double sumMain = 0.0;
        double sumBust = 0.0;
        double sum21   = 0.0;
        double sumPair = 0.0;
        double sumHot3 = 0.0;
        for (unsigned r = 0; r < rounds; ++r) {
            // fresh copy of shoe
            std::array<int,52> local = shoeBuffer;
            // draw initial four cards: p1,p2,d1,d2
            int p1 = drawCard(local, gen);
            int p2 = drawCard(local, gen);
            int d1 = drawCard(local, gen);
            int d2 = drawCard(local, gen);
            // convert to card lists for side bet evaluation later
            std::vector<int> dealer = {d1, d2};
            std::vector<int> player = {p1, p2};
            // check blackjacks
            bool pbj = (handValue(player) == 21 && player.size() == 2);
            bool dbj = (handValue(dealer) == 21 && dealer.size() == 2);
            double mainResult = 0.0;
            if (pbj && !dbj) {
                mainResult = 1.5;
            } else if (dbj && !pbj) {
                mainResult = -1.0;
            } else if (pbj && dbj) {
                mainResult = 0.0;
            } else {
                auto [ptotal, bet] = playPlayer(player, cardValue(getCardRank(d1)), local, gen);
                auto [dtotal, dlen] = playDealer(dealer, local, gen);
                if (ptotal > 21) {
                    mainResult = -bet;
                } else if (dtotal > 21) {
                    mainResult = bet;
                    if (dlen >= 3) {
                        sumBust += 1.0;
                    }
                } else if (ptotal > dtotal) {
                    mainResult = bet;
                } else if (ptotal < dtotal) {
                    mainResult = -bet;
                }
            }
            sumMain += mainResult;
            // Evaluate side bets using first two player cards and dealer upcard
            std::string cp1 = decodeCard(p1);
            std::string cp2 = decodeCard(p2);
            std::string cd1 = decodeCard(d1);
            // 21+3
            std::string sb21 = SideBets::evaluate21Plus3({cp1, cp2}, cd1);
            if (!sb21.empty()) sum21 += SideBets::PAYOUT_21PLUS3.at(sb21);
            // Pair
            std::string sbp = SideBets::evaluatePair({cp1, cp2});
            if (!sbp.empty()) sumPair += SideBets::PAYOUT_PAIR.at(sbp);
            // Hot3
            std::string sbh = SideBets::evaluateHot3({cp1, cp2}, cd1);
            if (!sbh.empty()) sumHot3 += SideBets::PAYOUT_HOT3.at(sbh);
        }
        return { sumMain, sumBust, sum21, sumPair, sumHot3 };
    }
};

/**
 * StrategyAdvisor takes simulation results, card counters and shoe
 * information to produce human‑readable advice.  It supports Kelly
 * criterion suggestions and warns about dealer bust probabilities if
 * Bayesian totals are supplied.
 */
class StrategyAdvisor {
public:
    struct Config {
        double mainEvThreshold       = 0.0;
        double sidebetThreshold      = 0.0;
        double insuranceZenCount     = 3.0;
        double dealerBustAlertThresh = 0.35;
        double bustWarningFloor      = 0.15;
        bool   liveBayes             = true;
        bool   kellyEnabled          = false;
        double kellyRisk             = 1.5;
    };

    explicit StrategyAdvisor(const Config &cfg = Config()) : cfg(cfg) {}

    std::vector<std::string> recommend(const Shoe &shoe,
                                       const ZenCount &zenCounter,
                                       const WongHalves &wongCounter,
                                       const std::map<std::string, double> &simResults,
                                       const std::map<std::string, double> *bayesTotals = nullptr) {
        std::vector<std::string> recommendations;
        double mainEV = simResults.count("main_ev") ? simResults.at("main_ev") : 0.0;
        double decksRem = shoe.decksRemaining();
        double zenTrue  = zenCounter.trueCount(decksRem);
        double wongTrue = wongCounter.trueCount(decksRem);
        double advantagePct = mainEV * 100.0;
        if (mainEV > cfg.mainEvThreshold) {
            recommendations.push_back("Main bet advantage: +" + formatPercent(advantagePct) + ". Recommend increasing bet.");
        } else {
            recommendations.push_back("Main bet advantage: " + formatPercent(advantagePct) + ". No advantage - bet minimum.");
        }
        // Kelly criterion
        if (cfg.kellyEnabled && mainEV > 0.0) {
            double kellyFraction = mainEV / (cfg.kellyRisk * cfg.kellyRisk);
            recommendations.push_back("Recommended Kelly bet size: " + formatPercent(kellyFraction * 100) + " of bankroll");
        }
        // Side bets
        auto addSide = [&](const std::string &name, double ev) {
            std::string msg = name + " Side Bet EV = " + formatDecimal(ev) + ". ";
            if (ev > cfg.sidebetThreshold) msg += "+EV! Consider betting.";
            else msg += "Not profitable to bet.";
            recommendations.push_back(msg);
        };
        addSide("21+3", simResults.count("21+3_ev") ? simResults.at("21+3_ev") : 0.0);
        addSide("Perfect Pair", simResults.count("pair_ev") ? simResults.at("pair_ev") : 0.0);
        addSide("Hot 3", simResults.count("hot3_ev") ? simResults.at("hot3_ev") : 0.0);
        addSide("Bust-O-Rama", simResults.count("bust_ev") ? simResults.at("bust_ev") : 0.0);
        // Bayesian totals
        if (bayesTotals) {
            std::vector<std::string> lines;
            lines.push_back("Dealer Bayesian Total Prediction:");
            for (const auto &kv : *bayesTotals) {
                std::string key = kv.first == "bust" ? "Bust" : kv.first;
                lines.push_back(key + ": " + formatPercent(kv.second * 100) + "%");
            }
            double bustChance = bayesTotals->count("bust") ? bayesTotals->at("bust") : 0.0;
            if (bustChance >= cfg.dealerBustAlertThresh) {
                lines.push_back("High dealer bust likelihood. Consider conservative plays or bust side bet.");
            } else if (bustChance <= cfg.bustWarningFloor) {
                lines.push_back("Low dealer bust likelihood—consider avoiding bust-based side bets.");
            }
            recommendations.insert(recommendations.end(), lines.begin(), lines.end());
        }
        return recommendations;
    }

private:
    Config cfg;
    std::string formatPercent(double value) const {
        char buf[32];
        std::snprintf(buf, sizeof(buf), "%.2f%%", value);
        return std::string(buf);
    }
    std::string formatDecimal(double value) const {
        char buf[32];
        std::snprintf(buf, sizeof(buf), "%.3f", value);
        return std::string(buf);
    }
};

/**
 * A simple demonstration entry point.  This main function runs a
 * simulation on a fresh eight‑deck shoe, prints expected values and
 * outputs some strategy recommendations for an example hand.  GUI
 * integration using Qt would replace this textual output.
 */
int main() {
    try {
        Shoe shoe(8);
        ZenCount zen;
        WongHalves wong;
        FastSimulator sim(shoe);
        // run a small simulation for demonstration
        auto results = sim.runBatch(2, 10000);
        std::cout << "Main EV: " << results["main_ev"] << "\n";
        std::cout << "21+3 EV: " << results["21+3_ev"] << "\n";
        std::cout << "Pair EV: " << results["pair_ev"] << "\n";
        std::cout << "Hot3 EV: " << results["hot3_ev"] << "\n";
        std::cout << "Bust EV: " << results["bust_ev"] << "\n";
        // Suppose player has AS and 8D and dealer upcard is 6C
        std::vector<std::string> player = {"AS", "8D"};
        std::string dealer = "6C";
        // update counts accordingly
        for (const auto &c : player) {
            shoe.removeCard(c);
            zen.update(c);
            wong.update(c);
        }
        shoe.removeCard(dealer);
        zen.update(dealer);
        wong.update(dealer);
        // compute true count
        double tc = zen.trueCount(shoe.decksRemaining());
        std::string advice = DecisionAdvisor::recommendAction(player, dealer, tc);
        std::cout << "Decision advice: " << advice << "\n";
        // compute Bayesian predictions
        auto bayes = BayesianPredictor::dealerTotalProbabilities(dealer, shoe.getRemainingCards(), 5000);
        StrategyAdvisor advisor;
        auto recs = advisor.recommend(shoe, zen, wong, results, &bayes);
        for (const auto &line : recs) {
            std::cout << line << "\n";
        }
    } catch (const std::exception &ex) {
        std::cerr << "Error: " << ex.what() << "\n";
    }
    return 0;
}