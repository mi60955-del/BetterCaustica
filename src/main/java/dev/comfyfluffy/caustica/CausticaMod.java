package dev.comfyfluffy.caustica;

import net.fabricmc.api.ModInitializer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public final class CausticaMod implements ModInitializer {
	public static final String MOD_ID = "bettercaustica";
	public static final Logger LOGGER = LoggerFactory.getLogger("Caustica");

	@Override
	public void onInitialize() {
		// saveIfMissing registers every setting (applying TOML file values) and writes a default
		// config on first run.
		CausticaConfig.saveIfMissing();
		LOGGER.info("Caustica initialized (common); config: {}", CausticaConfig.configPath());
	}
}
