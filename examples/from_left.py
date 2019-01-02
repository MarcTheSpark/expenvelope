from expenvelope import Envelope

discontinuous_envelope = Envelope.from_levels_and_durations((3, 4, -2, -5), (2, 0, 3), (1, 0, -5))
discontinuous_envelope.show_plot(resolution=300)

print(discontinuous_envelope.value_at(2))
print(discontinuous_envelope.value_at(2, from_left=True))
