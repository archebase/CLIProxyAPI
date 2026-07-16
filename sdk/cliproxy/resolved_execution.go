package cliproxy

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"strings"

	nativeexecutor "github.com/router-for-me/CLIProxyAPI/v7/internal/runtime/executor"
	coreauth "github.com/router-for-me/CLIProxyAPI/v7/sdk/cliproxy/auth"
	cliproxyexecutor "github.com/router-for-me/CLIProxyAPI/v7/sdk/cliproxy/executor"
	coreusage "github.com/router-for-me/CLIProxyAPI/v7/sdk/cliproxy/usage"
	sdktranslator "github.com/router-for-me/CLIProxyAPI/v7/sdk/translator"
)

type NativeExecutorKind string

const (
	NativeExecutorOpenAICompatible NativeExecutorKind = "openai-compatible"
	NativeExecutorAnthropic        NativeExecutorKind = "anthropic"
)

type ResolvedExecutionCandidate struct {
	Auth         *coreauth.Auth
	RuntimeModel string
	APIBase      string
}

type ResolvedExecutionRequest struct {
	Executor   NativeExecutorKind
	Provider   string
	Candidates []ResolvedExecutionCandidate
	Request    cliproxyexecutor.Request
	Options    cliproxyexecutor.Options
}

func (r *EmbeddedRuntime) ExecuteResolved(ctx context.Context, execution ResolvedExecutionRequest) (cliproxyexecutor.Response, error) {
	ctx, finish, err := r.beginResolvedExecution(ctx)
	if err != nil {
		return cliproxyexecutor.Response{}, err
	}
	defer finish()
	executor, provider, candidates, err := r.prepareResolvedCandidates(execution)
	if err != nil {
		return cliproxyexecutor.Response{}, err
	}
	var lastErr error
	for _, candidate := range candidates {
		req := execution.Request
		req.Model = candidate.RuntimeModel
		execCtx, execReq, execOpts := r.prepareResolvedRequest(ctx, candidate, req, execution.Options, execution.Executor, execution.Request.Model)
		resp, errExec := executor.Execute(execCtx, candidate.Auth, execReq, execOpts)
		if errExec == nil {
			return resp, nil
		}
		if err := ctx.Err(); err != nil {
			return cliproxyexecutor.Response{}, err
		}
		if resolvedRequestInvalid(errExec) {
			return cliproxyexecutor.Response{}, errExec
		}
		lastErr = errExec
	}
	if lastErr != nil {
		return cliproxyexecutor.Response{}, lastErr
	}
	return cliproxyexecutor.Response{}, &coreauth.Error{Code: "auth_not_found", Message: fmt.Sprintf("no %s auth available", provider)}
}

func (r *EmbeddedRuntime) ExecuteResolvedStream(ctx context.Context, execution ResolvedExecutionRequest) (*cliproxyexecutor.StreamResult, error) {
	ctx, finish, err := r.beginResolvedExecution(ctx)
	if err != nil {
		return nil, err
	}
	executor, provider, candidates, err := r.prepareResolvedCandidates(execution)
	if err != nil {
		finish()
		return nil, err
	}
	var lastErr error
	for _, candidate := range candidates {
		req := execution.Request
		req.Model = candidate.RuntimeModel
		execCtx, execReq, execOpts := r.prepareResolvedRequest(ctx, candidate, req, execution.Options, execution.Executor, execution.Request.Model)
		result, errExec := executor.ExecuteStream(execCtx, candidate.Auth, execReq, execOpts)
		if errExec == nil {
			first, ok := <-result.Chunks
			if ok && first.Err != nil {
				if resolvedRequestInvalid(first.Err) {
					finish()
					return nil, first.Err
				}
				lastErr = first.Err
				continue
			}
			forwarded := make(chan cliproxyexecutor.StreamChunk)
			go func() {
				defer close(forwarded)
				defer finish()
				if ok {
					select {
					case forwarded <- first:
					case <-ctx.Done():
						return
					}
				}
				for chunk := range result.Chunks {
					select {
					case forwarded <- chunk:
					case <-ctx.Done():
						return
					}
				}
			}()
			return &cliproxyexecutor.StreamResult{Headers: result.Headers, Chunks: forwarded}, nil
		}
		if err := ctx.Err(); err != nil {
			finish()
			return nil, err
		}
		if resolvedRequestInvalid(errExec) {
			finish()
			return nil, errExec
		}
		lastErr = errExec
	}
	finish()
	if lastErr != nil {
		return nil, lastErr
	}
	return nil, &coreauth.Error{Code: "auth_not_found", Message: fmt.Sprintf("no %s auth available", provider)}
}

type preparedResolvedCandidate struct {
	Auth          *coreauth.Auth
	TransportAuth *coreauth.Auth
	RuntimeModel  string
}

func (r *EmbeddedRuntime) prepareResolvedCandidates(execution ResolvedExecutionRequest) (coreauth.ProviderExecutor, string, []preparedResolvedCandidate, error) {
	if len(execution.Candidates) == 0 {
		return nil, "", nil, &coreauth.Error{Code: "auth_not_found", Message: "no resolved auth candidates"}
	}
	provider := strings.ToLower(strings.TrimSpace(execution.Provider))
	if provider == "" {
		provider = "resolved"
	}
	var executor coreauth.ProviderExecutor
	switch execution.Executor {
	case NativeExecutorOpenAICompatible:
		executor = nativeexecutor.NewOpenAICompatExecutor(provider, r.service.cfg)
	case NativeExecutorAnthropic:
		provider = "claude"
		executor = nativeexecutor.NewClaudeExecutor(r.service.cfg)
	default:
		return nil, "", nil, fmt.Errorf("cliproxy: unsupported resolved executor %q", execution.Executor)
	}
	sequence := r.resolvedSequence.Add(1)
	candidates := make([]preparedResolvedCandidate, 0, len(execution.Candidates))
	for index, candidate := range execution.Candidates {
		if candidate.Auth == nil {
			return nil, "", nil, fmt.Errorf("cliproxy: resolved auth candidate %d is required", index)
		}
		model := strings.TrimSpace(candidate.RuntimeModel)
		if model == "" {
			return nil, "", nil, fmt.Errorf("cliproxy: resolved runtime model is required")
		}
		apiBase := resolvedExecutorBase(execution.Executor, candidate.APIBase)
		if apiBase == "" {
			return nil, "", nil, fmt.Errorf("cliproxy: resolved API base is required")
		}
		apiKey := strings.TrimSpace(candidate.Auth.Attributes["api_key"])
		if apiKey == "" {
			return nil, "", nil, fmt.Errorf("cliproxy: resolved API key is required")
		}
		transportAuth := candidate.Auth.Clone()
		auth := candidate.Auth.Clone()
		if strings.TrimSpace(auth.ID) == "" {
			auth.ID = fmt.Sprintf("resolved-%p-%d-%d", r, sequence, index)
			transportAuth.ID = auth.ID
		}
		auth.Provider = provider
		auth.Status = coreauth.StatusActive
		if auth.Attributes == nil {
			auth.Attributes = make(map[string]string)
		}
		auth.Attributes["api_key"] = apiKey
		auth.Attributes["base_url"] = apiBase
		candidates = append(candidates, preparedResolvedCandidate{Auth: auth, TransportAuth: transportAuth, RuntimeModel: model})
	}
	return executor, provider, candidates, nil
}

func (r *EmbeddedRuntime) prepareResolvedRequest(ctx context.Context, candidate preparedResolvedCandidate, req cliproxyexecutor.Request, opts cliproxyexecutor.Options, kind NativeExecutorKind, requestedModel string) (context.Context, cliproxyexecutor.Request, cliproxyexecutor.Options) {
	ctx = coreusage.WithPublishingSuppressed(ctx)
	if r.manager != nil {
		if roundTripper := r.manager.RoundTripperFor(candidate.TransportAuth); roundTripper != nil {
			ctx = coreauth.WithRoundTripper(ctx, roundTripper)
		}
	}
	if opts.RequestAfterAuthInterceptor == nil {
		return ctx, req, opts
	}
	opts.Headers = opts.Headers.Clone()
	if opts.Headers == nil {
		opts.Headers = make(http.Header)
	}
	toFormat := sdktranslator.FormatOpenAI
	if kind == NativeExecutorAnthropic {
		toFormat = sdktranslator.FormatClaude
	}
	intercepted := opts.RequestAfterAuthInterceptor(ctx, cliproxyexecutor.RequestAfterAuthInterceptRequest{
		SourceFormat:   opts.SourceFormat,
		ToFormat:       toFormat,
		Model:          req.Model,
		RequestedModel: resolvedRequestedModel(opts, requestedModel),
		Stream:         opts.Stream,
		Headers:        opts.Headers.Clone(),
		Body:           append([]byte(nil), req.Payload...),
		Metadata:       cloneResolvedMetadata(opts.Metadata),
	})
	for _, key := range intercepted.ClearHeaders {
		opts.Headers.Del(key)
	}
	for key, values := range intercepted.Headers {
		opts.Headers.Del(key)
		for _, value := range values {
			opts.Headers.Add(key, value)
		}
	}
	if len(intercepted.Body) > 0 {
		req.Payload = append([]byte(nil), intercepted.Body...)
		opts.OriginalRequest = append([]byte(nil), intercepted.Body...)
	}
	return ctx, req, opts
}

func resolvedRequestedModel(opts cliproxyexecutor.Options, fallback string) string {
	fallback = strings.TrimSpace(fallback)
	if len(opts.Metadata) == 0 {
		return fallback
	}
	switch value := opts.Metadata[cliproxyexecutor.RequestedModelMetadataKey].(type) {
	case string:
		if requested := strings.TrimSpace(value); requested != "" {
			return requested
		}
	case []byte:
		if requested := strings.TrimSpace(string(value)); requested != "" {
			return requested
		}
	}
	return fallback
}

func cloneResolvedMetadata(metadata map[string]any) map[string]any {
	if metadata == nil {
		return nil
	}
	cloned := make(map[string]any, len(metadata))
	for key, value := range metadata {
		cloned[key] = value
	}
	return cloned
}

func (r *EmbeddedRuntime) beginResolvedExecution(ctx context.Context) (context.Context, func(), error) {
	if r == nil || r.service == nil || r.service.cfg == nil {
		return nil, nil, fmt.Errorf("cliproxy: embedded runtime is unavailable")
	}
	if ctx == nil {
		ctx = context.Background()
	}
	if err := ctx.Err(); err != nil {
		return nil, nil, err
	}
	r.mu.Lock()
	started, closed := r.started, r.closed
	r.mu.Unlock()
	if !started || closed {
		return nil, nil, fmt.Errorf("cliproxy: embedded runtime is not started")
	}
	r.resolvedMu.Lock()
	if r.resolvedClosing {
		r.resolvedMu.Unlock()
		return nil, nil, fmt.Errorf("cliproxy: embedded runtime is closing")
	}
	r.resolvedInflight++
	r.resolvedMu.Unlock()
	return ctx, r.finishResolvedExecution, nil
}

func (r *EmbeddedRuntime) finishResolvedExecution() {
	r.resolvedMu.Lock()
	r.resolvedInflight--
	if r.resolvedClosing && r.resolvedInflight == 0 && r.resolvedDrained != nil {
		close(r.resolvedDrained)
		r.resolvedDrained = nil
	}
	r.resolvedMu.Unlock()
}

func (r *EmbeddedRuntime) closeResolvedExecutions(ctx context.Context) error {
	r.resolvedMu.Lock()
	r.resolvedClosing = true
	if r.resolvedInflight == 0 {
		r.resolvedMu.Unlock()
		return nil
	}
	if r.resolvedDrained == nil {
		r.resolvedDrained = make(chan struct{})
	}
	drained := r.resolvedDrained
	r.resolvedMu.Unlock()
	select {
	case <-drained:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

func resolvedExecutorBase(kind NativeExecutorKind, apiBase string) string {
	apiBase = strings.TrimRight(strings.TrimSpace(apiBase), "/")
	if kind == NativeExecutorAnthropic {
		apiBase = strings.TrimSuffix(apiBase, "/v1")
	}
	return apiBase
}

func resolvedRequestInvalid(err error) bool {
	var scoped interface{ IsRequestScoped() bool }
	if errors.As(err, &scoped) && scoped.IsRequestScoped() {
		return true
	}
	var status interface{ StatusCode() int }
	if errors.As(err, &status) {
		return status.StatusCode() == http.StatusBadRequest || status.StatusCode() == http.StatusUnprocessableEntity
	}
	return false
}
