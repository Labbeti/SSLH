
def cache_feature(func):
	def decorator(*args, **kwargs):
		key = ",".join(map(str, args))

		if key not in decorator.cache:
			decorator.cache[key] = func(*args, **kwargs)

		return decorator.cache[key]

	decorator.cache = dict()
	return decorator
