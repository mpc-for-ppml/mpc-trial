import sys
from mpyc.runtime import mpc

m = len(mpc.parties)

if m%2 == 0:
    print('Odd number of parties required.')
    sys.exit()

t = m//2

voters = list(range(t+1))  # parties P[0],...,P[t]

if mpc.pid in voters:
    vote = int(sys.argv[1]) if sys.argv[1:] else 1  # default "yes"
else:
    vote = None  # no input

secbit = mpc.SecFld(2)  # secure bits over GF(2)

mpc.run(mpc.start())
votes = mpc.input(secbit(vote), senders=voters)
result = mpc.run(mpc.output(mpc.all(votes), receivers=voters))
mpc.run(mpc.shutdown())

if result is None:  # no output
    print('Thanks for serving as oblivious matchmaker;)')
elif result:
    print(f'Match: unanimous agreement between {t+1} part{"ies" if t else "y"}!')
else:
    print(f'No match: someone disagrees among {t+1} part{"ies" if t else "y"}?')