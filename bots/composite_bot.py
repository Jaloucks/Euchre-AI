class CompositeBot:
    def __init__(self, bidder, player):
        self.bidder = bidder
        self.player = player

    def choose_bid(self, view, legal_actions):
        return self.bidder.choose_bid(view, legal_actions)

    def choose_discard(self, view):
       return self.bidder.choose_discard(view)

    def choose_play(self, view):
       return self.player.choose_play(view)

    def as_callbacks(self):
        return self.choose_bid, self.choose_discard, self.choose_play