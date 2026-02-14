import { create } from 'zustand';

export type ModalType = 
  | 'nodeInspector' 
  | 'chat' 
  | 'relationEditor' 
  | 'settings' 
  | 'pasteSpecial'
  | null;

interface ModalState {
  // Current open modal
  activeModal: ModalType;
  
  // Modal data/payload
  modalData: any;
  
  // Actions
  openModal: (modal: ModalType, data?: any) => void;
  closeModal: () => void;
  
  // Derived state
  isAnyModalOpen: boolean;
}

export const useModalStore = create<ModalState>((set, get) => ({
  activeModal: null,
  modalData: null,
  
  openModal: (modal, data) => set({ 
    activeModal: modal, 
    modalData: data,
    isAnyModalOpen: true 
  }),
  
  closeModal: () => set({ 
    activeModal: null, 
    modalData: null,
    isAnyModalOpen: false 
  }),
  
  isAnyModalOpen: false
}));

export default useModalStore;
