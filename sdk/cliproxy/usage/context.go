package usage

import "context"

type suppressPublishingKey struct{}

func WithPublishingSuppressed(ctx context.Context) context.Context {
	if ctx == nil {
		ctx = context.Background()
	}
	return context.WithValue(ctx, suppressPublishingKey{}, true)
}

func PublishingSuppressed(ctx context.Context) bool {
	if ctx == nil {
		return false
	}
	value, _ := ctx.Value(suppressPublishingKey{}).(bool)
	return value
}
