import { useState } from 'react';
import {
  ANATOMY_NODES,
  getAnatomyNode,
  getBreadcrumbChain,
  registerDisplayName,
} from '../content/llmAnatomy';
import type { Register } from '../state/atlasState';
import { NodeLab, type NodeLabProps } from './NodeLab';

const REGISTERS: Register[] = ['simple', 'sufficient', 'complete'];

interface AnatomyInspectorProps extends Omit<NodeLabProps, 'node'> {
  nodeId: string;
  register: Register;
  telemetryLines: string[];
  onSelectNode: (id: string) => void;
  onChangeRegister: (r: Register) => void;
}

export function AnatomyInspector(props: AnatomyInspectorProps) {
  const [showOutline, setShowOutline] = useState(false);
  const node = getAnatomyNode(props.nodeId);
  const breadcrumbs = getBreadcrumbChain(props.nodeId);
  const childNodes = node.children
    .map((id) => ANATOMY_NODES[id])
    .filter((n): n is NonNullable<typeof n> => Boolean(n));

  return (
    <aside className="inspector-rail" aria-live="polite" aria-label="LLM Anatomy Inspector">
      {/* Sticky Breadcrumb + 3-Mode Explanation Switch */}
      <div className="inspector-top">
        <div className="crumb-bar">
          <nav className="crumb-trail" aria-label="Anatomy path">
            {breadcrumbs.map((b, i) => (
              <span key={b.id} className="crumb-item">
                {i > 0 && <span className="crumb-sep">/</span>}
                <button
                  type="button"
                  className={`crumb-link ${b.id === node.id ? 'is-current' : ''}`}
                  onClick={() => {
                    setShowOutline(false);
                    props.onSelectNode(b.id);
                  }}
                >
                  {b.shortTitle}
                </button>
              </span>
            ))}
          </nav>

          <div className="crumb-actions">
            {node.parentId && (
              <button
                type="button"
                className="mini-btn"
                onClick={() => {
                  setShowOutline(false);
                  props.onSelectNode(node.parentId!);
                }}
              >
                ↑ Up
              </button>
            )}
            <button
              type="button"
              className={`mini-btn ${showOutline ? 'is-active' : ''}`}
              onClick={() => setShowOutline((v) => !v)}
            >
              {showOutline ? 'Close Map' : 'Full Map'}
            </button>
          </div>
        </div>

        <div className="register-switch" role="group" aria-label="Explanation depth">
          {REGISTERS.map((r) => (
            <button
              key={r}
              type="button"
              className={`reg-btn ${props.register === r ? 'is-active' : ''}`}
              onClick={() => props.onChangeRegister(r)}
            >
              {registerDisplayName(r)}
            </button>
          ))}
        </div>
      </div>

      {/* Body: Either Full Recursive Tree Map OR Current Node Deep Dive */}
      <div className="inspector-scroll">
        {showOutline ? (
          <div className="outline-view">
            <div className="section-label">Complete LLM Anatomy (34 Components)</div>
            <TreeBranch
              nodeId="root"
              activeId={props.nodeId}
              depth={0}
              onPick={(id) => {
                props.onSelectNode(id);
                setShowOutline(false);
              }}
            />
          </div>
        ) : (
          <>
            <div className="node-header">
              <div className="node-meta-row">
                <span
                  className={`status-pill ${
                    node.status === 'live' ? 'is-live' : 'is-ext'
                  }`}
                >
                  {node.status === 'live' ? 'Live in Miniature Model' : 'Modern Frontier Extension'}
                </span>
                {node.shapeTag && (
                  <span className="shape-pill mono">{node.shapeTag}</span>
                )}
              </div>
              <h1 className="node-title">{node.title}</h1>
            </div>

            <p className="node-prose">{node.copy[props.register]}</p>

            {props.register === 'simple' && node.simplificationNote && (
              <div className="note-banner">{node.simplificationNote}</div>
            )}

            {props.telemetryLines.length > 0 && (
              <div className="telemetry-strip mono">
                {props.telemetryLines.map((line, i) => (
                  <div key={i}>{line}</div>
                ))}
              </div>
            )}

            <NodeLab
              node={node}
              client={props.client}
              manifest={props.manifest}
              tokenizer={props.tokenizer}
              baseline={props.baseline}
              ablated={props.ablated}
              ablation={props.ablation}
              layer={props.layer}
              head={props.head}
              showPreSoftmax={props.showPreSoftmax}
              hoveredTokenIdx={props.hoveredTokenIdx}
              replay={props.replay}
              paramName={props.paramName}
              paramRow={props.paramRow}
              paramCol={props.paramCol}
              temperature={props.temperature}
              topK={props.topK}
              topP={props.topP}
              busy={props.busy}
              onChangeAblation={props.onChangeAblation}
              onChangeSampling={props.onChangeSampling}
              onHoverToken={props.onHoverToken}
              onInspectScalar={props.onInspectScalar}
              onStepAutoregressive={props.onStepAutoregressive}
              onTogglePreSoftmax={props.onTogglePreSoftmax}
            />

            {childNodes.length > 0 && (
              <div className="subparts-section">
                <div className="section-label">
                  Inside {node.shortTitle} ({childNodes.length} subparts — click to expand)
                </div>
                <div className="subparts-list">
                  {childNodes.map((child) => (
                    <button
                      key={child.id}
                      type="button"
                      className="subpart-row"
                      onClick={() => props.onSelectNode(child.id)}
                    >
                      <div className="subpart-top">
                        <span className="subpart-name">{child.title}</span>
                        <span className="subpart-right">
                          {child.children.length > 0 && (
                            <span className="subpart-count mono">
                              {child.children.length} inside
                            </span>
                          )}
                          <span
                            className={`dot-indicator ${
                              child.status === 'live' ? 'is-live' : 'is-ext'
                            }`}
                            title={
                              child.status === 'live'
                                ? 'Live in model'
                                : 'Frontier extension'
                            }
                          />
                          <span className="subpart-arrow">›</span>
                        </span>
                      </div>
                      <div className="subpart-desc">{child.oneLiner}</div>
                    </button>
                  ))}
                </div>
              </div>
            )}
          </>
        )}
      </div>
    </aside>
  );
}

function TreeBranch({
  nodeId,
  activeId,
  depth,
  onPick,
}: {
  nodeId: string;
  activeId: string;
  depth: number;
  onPick: (id: string) => void;
}) {
  const node = ANATOMY_NODES[nodeId];
  if (!node) return null;

  return (
    <div>
      <button
        type="button"
        className={`tree-row ${node.id === activeId ? 'is-current' : ''}`}
        style={{ paddingLeft: 10 + depth * 14 }}
        onClick={() => onPick(node.id)}
      >
        <span className="tree-name">{node.shortTitle}</span>
        <span
          className={`dot-indicator ${
            node.status === 'live' ? 'is-live' : 'is-ext'
          }`}
        />
      </button>
      {node.children.map((cid) => (
        <TreeBranch
          key={cid}
          nodeId={cid}
          activeId={activeId}
          depth={depth + 1}
          onPick={onPick}
        />
      ))}
    </div>
  );
}
