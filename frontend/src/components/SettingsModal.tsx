import { useEffect, useState } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { graphApi } from '../services/api';
import type { AppSettings, AppSettingsUpdate, RelationStyle } from '../types/settings';
import type { NodeType, RelationshipType } from '../types/graph';

interface Props {
	onClose: () => void;
}

const NODE_TYPES: NodeType[] = ['Observation', 'Hypothesis', 'Source', 'Concept', 'Entity'];
const REL_TYPES: RelationshipType[] = ['SUPPORTS', 'CONTRADICTS', 'RELATES_TO', 'OBSERVED_IN', 'DISCUSSES'];

export default function SettingsModal({ onClose }: Props) {
	const queryClient = useQueryClient();

	const { data: settingsResponse } = useQuery({
		queryKey: ['settings'],
		queryFn: () => graphApi.getSettings().then((r) => r.data as AppSettings),
	});

	const initialSettings = settingsResponse;

	const [localNodeColors, setLocalNodeColors] = useState<Record<string, string>>(
		() => initialSettings?.node_colors || {}
	);
	const [localRelStyles, setLocalRelStyles] = useState<Record<string, RelationStyle>>(
		() => initialSettings?.relation_styles || {}
	);
	const [showEdgeLabels, setShowEdgeLabels] = useState<boolean>(initialSettings?.show_edge_labels ?? true);

	// Keep local state in sync when settings load or change
	useEffect(() => {
		if (!initialSettings) return;
		setLocalNodeColors({ ...initialSettings.node_colors });
		setLocalRelStyles({ ...initialSettings.relation_styles });
		setShowEdgeLabels(initialSettings.show_edge_labels);
	}, [initialSettings]);

	const updateMutation = useMutation({
		mutationFn: (payload: AppSettingsUpdate) => graphApi.updateSettings(payload).then((r) => r.data),
		onSuccess: () => {
			queryClient.invalidateQueries({ queryKey: ['settings'] });
			queryClient.invalidateQueries({ queryKey: ['graph'] });
			onClose();
		},
	});

	const handleSave = () => {
		const payload: AppSettingsUpdate = {
			show_edge_labels: showEdgeLabels,
			node_colors: localNodeColors,
			relation_styles: localRelStyles,
		};
		updateMutation.mutate(payload);
	};

	return (
		<div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50" onClick={onClose}>
			<div className="bg-white rounded-lg shadow-xl w-full max-w-3xl max-h-[80vh] flex flex-col" onClick={(e) => e.stopPropagation()}>
				<div className="px-6 py-4 border-b flex justify-between items-center">
					<h2 className="text-lg font-semibold text-gray-800">User Settings</h2>
					<button onClick={onClose} className="text-gray-500 hover:text-gray-700 text-xl leading-none">
						×
					</button>
				</div>

				<div className="p-6 space-y-8 overflow-y-auto flex-1 min-h-0">
					{/* Edge Labels */}
					<div>
						<h3 className="text-sm font-semibold text-gray-700 mb-3">Edges</h3>
						<label className="inline-flex items-center gap-2 text-sm">
							<input
								type="checkbox"
								checked={showEdgeLabels}
								onChange={(e) => setShowEdgeLabels(e.target.checked)}
								className="rounded"
							/>
							Show edge labels
						</label>
					</div>

					{/* Node Colors */}
					<div>
						<h3 className="text-sm font-semibold text-gray-700 mb-3">Node Colors</h3>
						<div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
							{NODE_TYPES.map((t) => (
								<div key={t} className="flex items-center justify-between gap-4 border rounded-md p-3">
									<div className="flex items-center gap-2">
										<div
											className="w-5 h-5 rounded border"
											style={{ backgroundColor: localNodeColors[t] || '#666' }}
											aria-label={`${t} color preview`}
										/>
										<span className="text-sm text-gray-700">{t}</span>
									</div>
									<input
										type="color"
										value={localNodeColors[t] || '#666666'}
										onChange={(e) =>
											setLocalNodeColors((prev) => ({
												...prev,
												[t]: e.target.value,
											}))
										}
										className="w-14 h-8 p-0 border rounded"
										aria-label={`${t} color`}
									/>
								</div>
							))}
						</div>
					</div>

					{/* Relation Styles */}
					<div>
						<h3 className="text-sm font-semibold text-gray-700 mb-3">Relation Styles</h3>
						<div className="space-y-3">
							{REL_TYPES.map((rt) => {
								const style = localRelStyles[rt] || { line_color: '#6B7280', width: 2, target_arrow_shape: 'triangle' };
								return (
									<div key={rt} className="border rounded-md p-3">
										<div className="flex items-center justify-between mb-3">
											<span className="text-sm font-medium text-gray-700">{rt}</span>
											<div className="flex items-center gap-3">
												<label className="text-xs text-gray-600">Line</label>
												<input
													type="color"
													value={style.line_color}
													onChange={(e) =>
														setLocalRelStyles((prev) => ({
															...prev,
															[rt]: { ...style, line_color: e.target.value },
														}))
													}
													className="w-12 h-8 p-0 border rounded"
													aria-label={`${rt} line color`}
												/>
												<label className="text-xs text-gray-600">Arrow</label>
												<input
													type="color"
													value={style.target_arrow_color || style.line_color}
													onChange={(e) =>
														setLocalRelStyles((prev) => ({
															...prev,
															[rt]: { ...style, target_arrow_color: e.target.value },
														}))
													}
													className="w-12 h-8 p-0 border rounded"
													aria-label={`${rt} arrow color`}
												/>
												<label className="text-xs text-gray-600">Width</label>
												<input
													type="number"
													min={1}
													max={12}
													value={style.width ?? 2}
													onChange={(e) =>
														setLocalRelStyles((prev) => ({
															...prev,
															[rt]: { ...style, width: Number(e.target.value) },
														}))
													}
													className="w-16 px-2 py-1 border rounded text-sm"
													aria-label={`${rt} width`}
												/>
												<select
													value={style.line_style || 'solid'}
													onChange={(e) =>
														setLocalRelStyles((prev) => ({
															...prev,
															[rt]: { ...style, line_style: e.target.value as RelationStyle['line_style'] },
														}))
													}
													className="px-2 py-1 border rounded text-sm"
													aria-label={`${rt} line style`}
												>
													<option value="solid">Solid</option>
													<option value="dashed">Dashed</option>
													<option value="dotted">Dotted</option>
												</select>
												<select
													value={style.target_arrow_shape || 'triangle'}
													onChange={(e) =>
														setLocalRelStyles((prev) => ({
															...prev,
															[rt]: { ...style, target_arrow_shape: e.target.value as RelationStyle['target_arrow_shape'] },
														}))
													}
													className="px-2 py-1 border rounded text-sm"
													aria-label={`${rt} arrow shape`}
												>
													<option value="triangle">Triangle</option>
													<option value="tee">Tee</option>
													<option value="none">None</option>
												</select>
											</div>
										</div>
									</div>
								);
							})}
						</div>
					</div>
				</div>

				<div className="px-6 py-4 border-t flex justify-end gap-3">
					<button onClick={onClose} className="px-4 py-2 text-gray-700 hover:bg-gray-100 rounded-md transition-colors">
						Cancel
					</button>
					<button
						onClick={handleSave}
						disabled={updateMutation.isPending}
						className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
					>
						{updateMutation.isPending ? 'Saving...' : 'Save Settings'}
					</button>
				</div>
			</div>
		</div>
	);
}


