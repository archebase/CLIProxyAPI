package cliproxy

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"reflect"
	"strings"
	"sync"

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
		normalizer := newResolvedProviderResponseNormalizer(execution.Provider, execution.Executor, cliproxyexecutor.ResponseFormatOrSource(execution.Options))
		execCtx, execReq, execOpts := r.prepareResolvedRequest(ctx, candidate, req, execution.Options, execution.Executor, execution.Request.Model)
		resp, errExec := executor.Execute(execCtx, candidate.Auth, execReq, execOpts)
		if errExec == nil {
			resp.Payload = normalizer.Normalize(resp.Payload)
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
		normalizer := newResolvedProviderResponseNormalizer(execution.Provider, execution.Executor, cliproxyexecutor.ResponseFormatOrSource(execution.Options))
		execCtx, execReq, execOpts := r.prepareResolvedRequest(ctx, candidate, req, execution.Options, execution.Executor, execution.Request.Model)
		result, errExec := executor.ExecuteStream(execCtx, candidate.Auth, execReq, execOpts)
		if errExec == nil {
			var first cliproxyexecutor.StreamChunk
			var ok bool
			select {
			case first, ok = <-result.Chunks:
			case <-ctx.Done():
				finish()
				return nil, ctx.Err()
			}
			if ok && first.Err != nil {
				if resolvedRequestInvalid(first.Err) {
					finish()
					return nil, first.Err
				}
				lastErr = first.Err
				continue
			}
			for ok {
				first.Payload = normalizer.Normalize(first.Payload)
				if len(first.Payload) > 0 {
					break
				}
				select {
				case first, ok = <-result.Chunks:
					if ok && first.Err != nil {
						finish()
						return nil, first.Err
					}
				case <-ctx.Done():
					finish()
					return nil, ctx.Err()
				}
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
				for {
					var chunk cliproxyexecutor.StreamChunk
					var more bool
					select {
					case chunk, more = <-result.Chunks:
						if !more {
							return
						}
					case <-ctx.Done():
						return
					}
					if chunk.Err != nil {
						select {
						case forwarded <- chunk:
						case <-ctx.Done():
						}
						return
					}
					chunk.Payload = normalizer.Normalize(chunk.Payload)
					if len(chunk.Payload) == 0 {
						continue
					}
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

type resolvedProviderResponseNormalizer struct {
	stripClaudeThinking bool
	thinkingIndexes     map[int64]struct{}
	suppressedIndexes   map[int64]struct{}
}

func newResolvedProviderResponseNormalizer(provider string, executor NativeExecutorKind, responseFormat sdktranslator.Format) *resolvedProviderResponseNormalizer {
	strip := strings.EqualFold(strings.TrimSpace(provider), "zhipu") && executor == NativeExecutorAnthropic && responseFormat == sdktranslator.FormatClaude
	return &resolvedProviderResponseNormalizer{stripClaudeThinking: strip, thinkingIndexes: make(map[int64]struct{}), suppressedIndexes: make(map[int64]struct{})}
}

func (n *resolvedProviderResponseNormalizer) Normalize(payload []byte) []byte {
	if n == nil || !n.stripClaudeThinking {
		return payload
	}
	return n.stripClaudeThinkingPayload(payload)
}

func (n *resolvedProviderResponseNormalizer) stripClaudeThinkingPayload(payload []byte) []byte {
	if len(payload) == 0 {
		return payload
	}
	jsonStart := 0
	jsonEnd := len(payload)
	if dataOffset := sseResolvedDataOffset(payload); dataOffset >= 0 {
		jsonStart = bytes.IndexByte(payload[dataOffset:], '{')
		if jsonStart < 0 {
			return payload
		}
		jsonStart += dataOffset
		lineEnd := bytes.IndexByte(payload[jsonStart:], '\n')
		if lineEnd >= 0 {
			jsonEnd = jsonStart + lineEnd
		}
	}
	jsonPayload := bytes.TrimSpace(payload[jsonStart:jsonEnd])
	var message map[string]any
	if err := json.Unmarshal(jsonPayload, &message); err != nil {
		return payload
	}
	changed := false
	eventType, _ := message["type"].(string)
	index := resolvedJSONInt64(message["index"])
	switch eventType {
	case "content_block_start":
		block, _ := message["content_block"].(map[string]any)
		blockType, _ := block["type"].(string)
		if blockType == "thinking" || blockType == "redacted_thinking" {
			n.thinkingIndexes[index] = struct{}{}
			n.suppressedIndexes[index] = struct{}{}
			return nil
		}
	case "content_block_delta":
		delta, _ := message["delta"].(map[string]any)
		deltaType, _ := delta["type"].(string)
		_, suppressed := n.thinkingIndexes[index]
		if suppressed || deltaType == "thinking_delta" || deltaType == "signature_delta" {
			n.thinkingIndexes[index] = struct{}{}
			return nil
		}
	case "content_block_stop":
		if _, suppressed := n.thinkingIndexes[index]; suppressed {
			delete(n.thinkingIndexes, index)
			return nil
		}
	}
	if _, hasIndex := message["index"]; hasIndex {
		message["index"] = index - n.suppressedBefore(index)
		changed = true
	}
	if content, ok := message["content"].([]any); ok {
		filtered := make([]any, 0, len(content))
		for _, item := range content {
			block, isBlock := item.(map[string]any)
			blockType, _ := block["type"].(string)
			if isBlock && (blockType == "thinking" || blockType == "redacted_thinking") {
				changed = true
				continue
			}
			filtered = append(filtered, item)
		}
		if changed {
			message["content"] = filtered
		}
	}
	if stopReason, ok := message["stop_reason"].(string); ok && strings.Contains(strings.ToLower(stopReason), "thinking") {
		message["stop_reason"] = "end_turn"
		changed = true
	}
	if delta, ok := message["delta"].(map[string]any); ok {
		if stopReason, ok := delta["stop_reason"].(string); ok && strings.Contains(strings.ToLower(stopReason), "thinking") {
			delta["stop_reason"] = "end_turn"
			changed = true
		}
	}
	if !changed {
		return payload
	}
	cleaned, err := json.Marshal(message)
	if err != nil {
		return payload
	}
	out := make([]byte, 0, len(payload)-len(jsonPayload)+len(cleaned))
	out = append(out, payload[:jsonStart]...)
	out = append(out, cleaned...)
	out = append(out, payload[jsonEnd:]...)
	return out
}

func (n *resolvedProviderResponseNormalizer) suppressedBefore(index int64) int64 {
	var count int64
	for suppressed := range n.suppressedIndexes {
		if suppressed < index {
			count++
		}
	}
	return count
}

func resolvedJSONInt64(value any) int64 {
	switch typed := value.(type) {
	case float64:
		return int64(typed)
	case json.Number:
		parsed, _ := typed.Int64()
		return parsed
	default:
		return 0
	}
}

func sseResolvedDataOffset(payload []byte) int {
	for offset := 0; offset < len(payload); {
		lineEnd := bytes.IndexByte(payload[offset:], '\n')
		if lineEnd < 0 {
			lineEnd = len(payload) - offset
		}
		line := bytes.TrimSpace(payload[offset : offset+lineEnd])
		if bytes.HasPrefix(line, []byte("data:")) {
			return offset
		}
		offset += lineEnd + 1
	}
	return -1
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
		cloned[key] = cloneResolvedMetadataValue(value)
	}
	return cloned
}

func cloneResolvedMetadataValue(value any) any {
	cloned := cloneResolvedReflectValue(reflect.ValueOf(value))
	if !cloned.IsValid() {
		return nil
	}
	return cloned.Interface()
}

func cloneResolvedReflectValue(value reflect.Value) reflect.Value {
	if !value.IsValid() {
		return reflect.Value{}
	}
	switch value.Kind() {
	case reflect.Interface:
		if value.IsNil() {
			return reflect.Zero(value.Type())
		}
		cloned := cloneResolvedReflectValue(value.Elem())
		wrapped := reflect.New(value.Type()).Elem()
		wrapped.Set(cloned)
		return wrapped
	case reflect.Pointer:
		if value.IsNil() {
			return reflect.Zero(value.Type())
		}
		cloned := reflect.New(value.Type().Elem())
		cloned.Elem().Set(cloneResolvedReflectValue(value.Elem()))
		return cloned
	case reflect.Map:
		if value.IsNil() {
			return reflect.Zero(value.Type())
		}
		cloned := reflect.MakeMapWithSize(value.Type(), value.Len())
		iterator := value.MapRange()
		for iterator.Next() {
			cloned.SetMapIndex(cloneResolvedReflectValue(iterator.Key()), cloneResolvedReflectValue(iterator.Value()))
		}
		return cloned
	case reflect.Slice:
		if value.IsNil() {
			return reflect.Zero(value.Type())
		}
		cloned := reflect.MakeSlice(value.Type(), value.Len(), value.Len())
		for index := 0; index < value.Len(); index++ {
			cloned.Index(index).Set(cloneResolvedReflectValue(value.Index(index)))
		}
		return cloned
	case reflect.Array:
		cloned := reflect.New(value.Type()).Elem()
		for index := 0; index < value.Len(); index++ {
			cloned.Index(index).Set(cloneResolvedReflectValue(value.Index(index)))
		}
		return cloned
	case reflect.Struct:
		cloned := reflect.New(value.Type()).Elem()
		cloned.Set(value)
		for index := 0; index < value.NumField(); index++ {
			if cloned.Field(index).CanSet() && value.Field(index).CanInterface() {
				cloned.Field(index).Set(cloneResolvedReflectValue(value.Field(index)))
			}
		}
		return cloned
	default:
		return value
	}
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
	runtimeContext := r.resolvedContext
	r.resolvedMu.Unlock()
	execContext, cancel := context.WithCancel(ctx)
	stopRuntimeCancel := context.AfterFunc(runtimeContext, cancel)
	var finishOnce sync.Once
	finish := func() {
		finishOnce.Do(func() {
			stopRuntimeCancel()
			cancel()
			r.finishResolvedExecution()
		})
	}
	return execContext, finish, nil
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
	resolvedCancel := r.resolvedCancel
	if resolvedCancel != nil {
		resolvedCancel()
	}
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
